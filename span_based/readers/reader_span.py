import logging
from typing import List, Dict, Tuple
import json
from overrides import overrides
import numpy as np

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    TextField,
    SpanField,
    SequenceLabelField,
    ListField,
    MetadataField,
)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans

from allennlp.data.tokenizers.token import Token
from allennlp.data.fields.field import Field

from span_based.utils.domain_utils import DomainUtils

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("span_based_reader")
class BertClassificationReader(DatasetReader):

    def __init__(
        self,
        is_weak_supervision: bool,
        domain_utils: DomainUtils,
        lazy: bool = False,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {}
        self.tokenizer = tokenizer or SpacyTokenizer()
        self._domain_utils = domain_utils
        self._is_weak_supervision = is_weak_supervision


    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                line = line.strip("\n")
                line = json.loads(line)
                if not line:
                    continue
                question = line["question"]
                program = line["program"] if "program" in line else None
                scene = line["scene"] if "scene" in line else None
                answer = line["answer"] if "answer" in line else None

                if not question:
                    continue

                input = question
                is_gold_spans = "gold_spans" in line

                if is_gold_spans:
                    gold_spans = line["gold_spans"]
                    gold_spans_dict: Dict[Tuple[int, int], str] = {}
                    for span in gold_spans:
                        span_limits = (span['span'][0]+1, span['span'][1]+1)  # shift span by 1 to handle [CLS] token
                        gold_spans_dict[span_limits] = span['type']
                else:
                    gold_spans_dict = None

                yield self.text_to_instance(input, gold_spans_dict, scene, answer, program)

    def _get_wordpieces(self, text):  # should match the reader method
        tokens = self.tokenizer.tokenize(text)
        do_lowercase = True
        tokens_out = (
            token.text.lower()
            if do_lowercase and token.text not in self.tokenizer.never_split
            else token.text
            for token in tokens
        )
        wps = [
            [wordpiece for wordpiece in self._token_indexers['bert'].wordpiece_tokenizer(token)]
            for token in tokens_out
        ]
        wps_flat = [wordpiece for token in wps for wordpiece in token]
        return tuple(wps_flat)

    @overrides
    def text_to_instance(self, source_string: str, gold_spans: Dict[Tuple[int, int], str],
                         scene_string: str, answer: str, program: str ) -> Instance:  # type: ignore
        """Turns raw source string and target string into an ``Instance``."""
        tokens = self.tokenizer.tokenize(source_string)
        word_pieces = self._get_wordpieces(source_string)
        word_pieces_tokens = [Token('[CLS]')] + [Token(wp) for wp in word_pieces] + [Token('[SEP]')]

        text_field = TextField(tokens, self._token_indexers)
        wp_field = TextField(word_pieces_tokens, self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}

        if gold_spans is None:
            constants = self._domain_utils.get_constants(program)

        spans: List[Field] = []
        gold_labels = []

        for start, end in enumerate_spans(word_pieces):
            # Shift by 1 due to CLS token
            spans.append(SpanField(start + 1, end + 1, wp_field))

            if gold_spans is not None:
                # Shift by 1 due to CLS token
                gold_labels.append(gold_spans.get((start + 1, end + 1), "NO-LABEL"))
            else:
                # Create random labels for each span so that labels would be collected. When no
                # more true labels are left, draw between NO-LABEL and span. These randomly assigned
                # labels would be ignored during training
                if constants[0]:
                    gold_labels.append(constants[0].pop())
                else:
                    rand_label = np.random.choice(a=["NO-LABEL", "span"], size=1, p=[0.7, 0.3])
                    gold_labels.append(rand_label[0])

        span_list_field: ListField = ListField(spans)
        fields["spans"] = span_list_field

        fields["span_labels"] = SequenceLabelField(
            gold_labels,
            span_list_field,
            label_namespace="labels",
        )

        metadata = {"tokens": word_pieces, "scene_str": scene_string, "answer": answer}
        if program:
            metadata["program"] = program
        if gold_spans:
            metadata["gold_spans"] = gold_spans

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)
