from typing import List, Dict, Any

from overrides import overrides

from allennlp.training.metrics import Metric


@Metric.register("sets_f1")
class SetsF1(Metric):
    """
    Simple metric to calculate F1 between a gold set of spans and a prdicted one.
    """

    def __init__(self) -> None:
        self._true_positive_counts = 0.
        self._total_gold_counts = 0.
        self._total_predicted_counts = 0.

    @overrides
    def reset(self) -> None:
        self._true_positive_counts = 0.
        self._total_gold_counts = 0.
        self._total_predicted_counts = 0.

    @overrides
    def __call__(self,
                 predicted_spans: List[Any],
                 gold_spans: List[Any]) -> None:
        for predicted, gold in zip(predicted_spans, gold_spans):
            self._total_gold_counts += len(gold)
            self._total_predicted_counts += len(predicted)
            shared_spans = {k: predicted[k] for k in predicted if k in
                            gold and predicted[k] == gold[k]}
            self._true_positive_counts += len(shared_spans)

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if self._total_predicted_counts == 0 or self._total_gold_counts == 0:
            f1 = 0.
        else:
            precision = self._true_positive_counts / self._total_predicted_counts
            recall = self._true_positive_counts / self._total_gold_counts
            f1 = 2 * precision * recall / (precision + recall)

        if reset:
            self.reset()

        return {"spans_f1": f1}
