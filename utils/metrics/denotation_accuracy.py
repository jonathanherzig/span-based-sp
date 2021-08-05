import gc
from typing import List, Dict
from overrides import overrides
from allennlp.training.metrics import Metric
from utils.executor import Executor


@Metric.register("denotation_accuracy")
class DenotationAccuracy(Metric):
    """
    Denotation accuracy based on program executions.
    """

    def __init__(self, executor: Executor) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.
        self._batch_counts = 0
        self._executor = executor

    @overrides
    def reset(self) -> None:
        self._correct_counts = 0.
        self._total_counts = 0.

    @overrides
    def __call__(self, predictions: List[List[str]], gold_targets: List[List[str]], scenes: List[str], answers: List[str], questions: List[List[str]]) -> None:
        self._total_counts += len(predictions)
        self._batch_counts += 1
        if self._batch_counts % 1000 == 0:  # collect garbage once in a while
            gc.collect()

        is_should_print = False
        is_printed = False
        for i, (predicted_tokens, scene, answer, question) in enumerate(zip(predictions, scenes, answers, questions)):

            gold_tokens = gold_targets[i] if gold_targets is not None else ['no_targets']

            for predicted in predicted_tokens:
                denotation = self._executor.execute(' '.join(predicted), scene)
                if not denotation.startswith('error_parse:'):
                    break

            gold_answer = answer if answer is not None else self._executor.execute(' '.join(gold_tokens))

            if gold_answer == denotation:
                self._correct_counts += 1
            elif not is_printed and is_should_print: # print errors but not too much
                print('ques: {}'.format(' '.join(question)))
                print('pred: {}'.format(' '.join(predicted)))
                print('gold: {}'.format(' '.join(gold_tokens)))
                print('deno: {}'.format(denotation))
                print('answ: {}'.format(gold_answer))
                print()
                is_printed = True

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if self._total_counts == 0:
            accuracy = 0.
        else:
            accuracy = self._correct_counts / self._total_counts

        if reset:
            self.reset()

        return {"den_acc": accuracy}
