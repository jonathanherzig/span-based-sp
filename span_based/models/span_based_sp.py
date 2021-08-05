from typing import Dict, Union, List, NamedTuple, Any

import torch
from torch.nn.modules.linear import Linear
from span_based.models.modeling_bert_span import BertModel

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, Metric
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.modules.span_extractors.span_extractor import SpanExtractor
from allennlp.modules import TimeDistributed, FeedForward

from treelib import Tree
from span_based.utils.translate_span_trees_to_programs import TreeMapper
from span_based.utils.domain_utils import DomainUtils
from span_based.utils.cky import CKY, Grammar
from span_based.utils.zero_shot_extractor import ZeroShotExtractor
from span_based.utils.weak_supervision_accuracy_metric import WeakSupervisionAccuracy
from span_based.utils.f1_sets_metric import SetsF1
from span_based.utils.spans_to_label_matrix import LabelsPreparer


class SpanInformation(NamedTuple):
    """
    A helper namedtuple for handling decoding information.
    # Parameters
    start : `int`
        The start index of the span.
    end : `int`
        The exclusive end index of the span.
    no_label_prob : `float`
        The probability of this span being assigned the `NO-LABEL` label.
    label_prob : `float`
        The probability of the most likely label.
    """

    start: int
    end: int
    label_prob: float
    no_label_prob: float
    label_index: int


@Model.register("span_based_sp")
class SpanBasedSP(Model):
    """
    The Span-based SP model that runs pretrained BERT,
    takes the pooled output, and creates span representation on top of it
    for classification.

    Based on AllenNLP bert_for_classification model.

    # Parameters

    vocab : `Vocabulary`
    bert_model : `Union[str, BertModel]`
        The BERT model to be wrapped. If a string is provided, we will call
        `BertModel.from_pretrained(bert_model)` and use the result.
    num_labels : `int`, optional (default: None)
        How many output classes to predict. If not provided, we'll use the
        vocab_size for the `label_namespace`.
    index : `str`, optional (default: "bert")
        The index of the token indexer that generates the BERT indices.
    label_namespace : `str`, optional (default : "labels")
        Used to determine the number of classes if `num_labels` is not supplied.
    trainable : `bool`, optional (default : True)
        If True, the weights of the pretrained BERT model will be updated during training.
        Otherwise, they will be frozen and only the final linear layer will be trained.
    initializer : `InitializerApplicator`, optional
        If provided, will be used to initialize the final linear layer *only*.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        bert_model: Union[str, BertModel],
        span_extractor: SpanExtractor,
        tree_mapper: TreeMapper,
        domain_utils: DomainUtils,
        is_weak_supervision: bool,
        feedforward: FeedForward = None,
        dropout: float = 0.0,
        num_labels: int = None,
        index: str = "bert",
        label_namespace: str = "labels",
        trainable: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        denotation_based_metric: Metric = None,
        token_based_metric: Metric = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        if isinstance(bert_model, str):
            self.bert_model = PretrainedBertModel.load(bert_model)
        else:
            self.bert_model = bert_model

        for param in self.bert_model.parameters():
            param.requires_grad = trainable

        in_features = self.bert_model.config.hidden_size

        self._label_namespace = label_namespace

        self.span_extractor = span_extractor
        self.feedforward_layer = TimeDistributed(feedforward) if feedforward else None
        self.num_classes = self.vocab.get_vocab_size("labels")
        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = span_extractor.get_output_dim()
        self.tag_projection_layer = TimeDistributed(Linear(output_dim, self.num_classes))

        if num_labels:
            out_features = num_labels
        else:
            out_features = vocab.get_vocab_size(namespace=self._label_namespace)

        self._dropout = torch.nn.Dropout(p=dropout)

        self._tree_mapper = tree_mapper

        labels = self.vocab.get_index_to_token_vocabulary(self._label_namespace)
        grammar = Grammar(labels)
        self._cky = CKY(grammar, tree_mapper, domain_utils)

        use_lexicon = True
        if use_lexicon:
            self.zero_shot_extractor = ZeroShotExtractor(labels, domain_utils)
            self._sim_weight = torch.nn.Parameter(
                torch.ones([1], dtype=torch.float32, requires_grad=True))

        self._classification_layer = torch.nn.Linear(in_features, out_features)
        self._accuracy = CategoricalAccuracy()
        self._accuracy_all_no_span = CategoricalAccuracy()
        self._fmeasure = F1Measure(positive_label=1)
        self._denotation_based_metric = denotation_based_metric
        self._token_based_metric = token_based_metric
        self._loss = torch.nn.CrossEntropyLoss()
        self._index = index
        initializer(self._classification_layer)

        self._epoch_counter = 0

        self._is_weak_supervision = is_weak_supervision
        if self._is_weak_supervision:
            self._weak_supervision_acc = WeakSupervisionAccuracy()
            self._label_preparer = LabelsPreparer(self.vocab.get_index_to_token_vocabulary(self._label_namespace))

        self._sets_f1_metric = SetsF1()
        self._compute_spans_f1 = False

    def forward(  # type: ignore
        self,
        tokens: TextFieldTensors,
        spans: torch.Tensor,
        metadata: List[Dict[str, Any]],
        span_labels: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : TextFieldTensors
            From a `TextField` (that has a bert-pretrained token indexer)
        label : torch.IntTensor, optional (default = None)
            From a `LabelField`

        # Returns

        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape `(batch_size, num_labels)` representing
            unnormalized log probabilities of the label.
        probs : torch.FloatTensor
            A tensor of shape `(batch_size, num_labels)` representing
            probabilities of the label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        inputs = tokens[self._index]
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = (input_ids != 0).long()

        encoded_text, pooled, *_ = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=mask
        )

        # Looking at the span start index is enough to know if
        # this is padding or not. Shape: (batch_size, num_spans)
        span_mask = (spans[:, :, 0] >= 0).squeeze(-1).long()
        if span_mask.dim() == 1:
            # This happens if you use batch_size 1 and encounter
            # a length 1 sentence in PTB, which do exist. -.-
            span_mask = span_mask.unsqueeze(-1)
        if span_labels is not None and span_labels.dim() == 1:
            span_labels = span_labels.unsqueeze(-1)

        num_spans = get_lengths_from_binary_sequence_mask(span_mask)

        span_representations = self.span_extractor(encoded_text, spans, mask, span_mask)

        if self.feedforward_layer is not None:
            span_representations = self.feedforward_layer(span_representations)

        logits = self.tag_projection_layer(span_representations)
        if self.zero_shot_extractor:
            alpha = 10.0  # hyper-param for boosting zero-shot weights
            # binary features for having a span match a lexicon entry.
            zero_shot_features = self.zero_shot_extractor.get_similarity_features(
                [meta['tokens'] for meta in metadata], spans)
            logits = logits + self._sim_weight * zero_shot_features * alpha

        input_tokens = [meta["tokens"] for meta in metadata]
        target_tokens = [meta['program'].split() for meta in metadata] if (
                len(metadata) > 0 and "program" in metadata[0]) else None

        output_dict = {
            "spans": spans,
            "tokens": input_tokens,
            "num_spans": num_spans,
        }

        if self.training and self._is_weak_supervision:
            cky_output_train = self.construct_trees(
                spans.data,
                input_tokens,
                logits.cpu().data,
                target_tokens,
            )
            predicted_trees_train = [output[0] for output in cky_output_train]
            # cases where the model finds span trees the can be transformed to the gold program.
            success_indices = [i for i, tree in enumerate(predicted_trees_train) if
                               tree is not None]
            success = len(success_indices)
            self._weak_supervision_acc(success, len(predicted_trees_train))
            if success > 0:
                collected_spans = [output[1] for output in cky_output_train]
                # prepare labels that correspond to tree found by hard EM
                span_labels_from_cky = self._label_preparer.prepare_labels(
                    [collected_span for collected_span in collected_spans if collected_span is not None],
                    spans[success_indices, :])
                logits = logits[success_indices, :]
                span_labels = span_labels_from_cky
                span_mask = span_mask[success_indices, :]
                # important for calculating the average loss over the batch
                output_dict["effective_batch_size"] = success
            else:
                return {'loss': None}

        loss = sequence_cross_entropy_with_logits(logits, span_labels, span_mask)
        output_dict["loss"] = loss

        if not self.training and self._denotation_based_metric:
            cky_output_inference = self.construct_trees(
                spans.cpu().data,
                input_tokens,
                logits.cpu().data,
                target_tokens=[None]*len(input_tokens),
            )
            predicted_trees = [output[0] for output in cky_output_inference]
            predicted_lfs = []
            for tree in predicted_trees:
                if not tree:
                    predicted_lfs.append('_cannot_parse_')
                else:
                    try:
                        predicted_lfs.append(self._tree_mapper.map_tree_to_program(tree))
                    except:
                        predicted_lfs.append('_cannot_parse_')

            predicted_tokens = [[z.split()] for z in predicted_lfs]
            # get target tokens only in case there are any
            target_tokens = [meta['program'].split() for meta in metadata] if (
                        len(metadata) > 0 and "program" in metadata[0]) else None
            self._denotation_based_metric(predicted_tokens,
                                          target_tokens, [x["scene_str"] for x in metadata],
                                          [x["answer"] for x in metadata], [x["tokens"] for x in metadata])
            self._token_based_metric([x[0] for x in predicted_tokens], target_tokens)

            if 'gold_spans' in metadata[0]:  # get F1 metric for tree correctness
                self._compute_spans_f1 = True
                gold_spans = [x['gold_spans'] for x in metadata]
                collected_spans_infer = [output[1] for output in cky_output_inference]
                predicted_spans = [{(cs.span[0], cs.span[1]-1): cs.category for cs in
                                    collected_spans_example if cs.category != 'NO-LABEL'}
                                   if collected_spans_example is not None else None for
                                   collected_spans_example in collected_spans_infer]
                # failures have no spans
                predicted_spans = [[] if v is None else v for v in predicted_spans]
                self._sets_f1_metric(predicted_spans, gold_spans)

        return output_dict

    def construct_trees(
        self,
        all_spans: torch.LongTensor,
        sentences: List[List[str]],
        logits,
        target_tokens,
    ) -> List[Tree]:
        """
        Construct `treelib.Tree` as the span tree for each batch element by running CKY for search.
        """
        # Switch to using exclusive end spans.
        exclusive_end_spans = all_spans.clone()
        exclusive_end_spans[:, :, -1] += 1

        trees: List[Tree] = []
        for batch_index, (spans, sentence, logit, target_token) in enumerate(
                zip(exclusive_end_spans, sentences, logits, target_tokens)
        ):
            try:
                if target_token:
                    program = ' '.join(target_token)
                else:
                    program = None
                tree = self._cky.find_best_span_tree(sentence, logit, spans, gold_program=program)
            except Exception as e:
                print('cannot parse to tree {}'.format(e))
                tree = (None, None)
            trees.append(tree)
        return trees

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if reset and self.training:
            self._epoch_counter += 1
        metrics = {}
        if self._compute_spans_f1:
            metrics.update(self._sets_f1_metric.get_metric(reset))
        if self._is_weak_supervision:
            metrics.update({"weak_sup_acc": self._weak_supervision_acc.get_metric(reset=reset)})
        if self._denotation_based_metric is not None:
            metrics.update(self._denotation_based_metric.get_metric(reset))
        if self._token_based_metric is not None:
            metrics.update(self._token_based_metric.get_metric(reset=reset))
        return metrics
