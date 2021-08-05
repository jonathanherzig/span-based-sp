"""Prepares labels to train on for a span tree found using hard EM."""

import torch


class LabelsPreparer:

    def __init__(self, labels):
        self._labels_inverse = {value: key for key, value in labels.items()}
        self._is_cude = torch.cuda.device_count() > 0

    def prepare_labels(self, spans_to_add, spans):
        """Gets found spans and returns an appropriate label tensor."""
        exclusive_end_spans = spans.clone()
        exclusive_end_spans[:, :, -1] += 1
        spans_shape = exclusive_end_spans.shape

        device = 'cuda' if self._is_cude else 'cpu'
        labels_for_model = torch.zeros([spans_shape[0], spans_shape[1]], dtype=torch.int64,
                                       requires_grad=False, device=device)
        for i, spans_to_add_example in enumerate(spans_to_add):
            spans_list = exclusive_end_spans[i, :].tolist()
            span_to_index = {}
            for j, span in enumerate(spans_list):
                start, end = span
                span_tup = (start, end)
                span_to_index[span_tup] = j
            for span_to_add in spans_to_add_example:
                span = span_to_add.span
                category = span_to_add.category
                labels_for_model[i, span_to_index[span]] = self._labels_inverse[category]
        return labels_for_model
