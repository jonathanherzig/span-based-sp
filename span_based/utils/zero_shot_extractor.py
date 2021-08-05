from collections import defaultdict

import torch

from allennlp.data.tokenizers.pretrained_transformer_pre_tokenizer import BertPreTokenizer
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer


class ZeroShotExtractor:

    def __init__(self, labels, domain_utils):
        self._tokenizer = BertPreTokenizer()  # should align with reader's tokenizer
        self._token_indexers = PretrainedBertIndexer(
            pretrained_model='bert-base-uncased')  # should align with reader's tokenizer
        self._is_cude = torch.cuda.device_count() > 0
        self._domain_utils = domain_utils
        self._set_labels_wordpieces(labels)

    def _get_wordpieces(self, text): # should match the reader method
        tokens = self._tokenizer.tokenize(text)
        do_lowercase = True
        tokens_out = (
            token.text.lower()
            if do_lowercase and token.text not in self._tokenizer.never_split
            else token.text
            for token in tokens
        )
        wps = [
            [wordpiece for wordpiece in self._token_indexers.wordpiece_tokenizer(token)]
            for token in tokens_out
        ]
        wps_flat = [wordpiece for token in wps for wordpiece in token]
        return tuple(wps_flat)

    def _set_labels_wordpieces(self, labels):
        self._num_labels = len(labels)
        self._labels_wordpieces = defaultdict(list)
        for index, label in labels.items():
            if label == 'NO-LABEL' or label == 'span':
                continue
            lexicon_phrases = self._domain_utils.get_lexicon_phrase(label)
            for lexicon_phrase in lexicon_phrases:
                self._labels_wordpieces[self._get_wordpieces(lexicon_phrase)].append(index)

    def get_similarity_features(self, batch_tokens, batch_spans):
        device = 'cuda' if self._is_cude else 'cpu'
        similarities = torch.zeros([batch_spans.shape[0], batch_spans.shape[1], self._num_labels], dtype=torch.float32,
                                   requires_grad=False, device=device)

        for k, (sentence, spans) in enumerate(zip(batch_tokens, batch_spans)):
            sent_len = len(sentence)
            span_to_ind = {}
            for i, span in enumerate(spans):
                span_to_ind[tuple(span.tolist())] = i
            for i in range(sent_len):
                for j in range(i+1, i+6):
                    if j > sent_len:
                        break
                    labels = self._labels_wordpieces.get(sentence[i:j])
                    if labels:
                        start = i + 1
                        end = j
                        for label in labels:
                            similarities[k, span_to_ind[(start, end)], label] = 1.0
        return similarities