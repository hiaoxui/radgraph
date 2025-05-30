from typing import Optional

import torch
import torch.distributed as dist

from radgraph.allennlp.common.util import is_distributed
from radgraph.allennlp.common.checks import ConfigurationError
from radgraph.allennlp.training.metrics.metric import Metric


@Metric.register("sequence_accuracy")
class SequenceAccuracy(Metric):
    """
    Sequence Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    """

    def __init__(self) -> None:
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, k, sequence_length).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, sequence_length).
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        """
        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)
        device = gold_labels.device

        # Some sanity checks.
        if gold_labels.dim() != predictions.dim() - 1:
            raise ConfigurationError(
                "gold_labels must have dimension == predictions.dim() - 1 but "
                "found tensor of shape: {}".format(gold_labels.size())
            )
        if mask is not None and mask.size() != gold_labels.size():
            raise ConfigurationError(
                "mask must have the same size as predictions but "
                "found tensor of shape: {}".format(mask.size())
            )

        k = predictions.size()[1]
        expanded_size = list(gold_labels.size())
        expanded_size.insert(1, k)
        expanded_gold = gold_labels.unsqueeze(1).expand(expanded_size)

        if mask is not None:
            expanded_mask = mask.unsqueeze(1).expand(expanded_size)
            masked_gold = expanded_mask * expanded_gold
            masked_predictions = expanded_mask * predictions
        else:
            masked_gold = expanded_gold
            masked_predictions = predictions

        eqs = masked_gold.eq(masked_predictions)
        matches_per_question = eqs.min(dim=2)[0]
        some_match = matches_per_question.max(dim=1)[0]
        correct = some_match.sum().item()

        _total_count = predictions.size()[0]
        _correct_count = correct

        if is_distributed():
            correct_count = torch.tensor(_correct_count).to(device)
            total_count = torch.tensor(_total_count).to(device)
            dist.all_reduce(correct_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            _correct_count = correct_count.item()
            _total_count = total_count.item()

        self.correct_count += _correct_count
        self.total_count += _total_count

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated accuracy.
        """
        if self.total_count > 0:
            accuracy = self.correct_count / self.total_count
        else:
            accuracy = 0
        if reset:
            self.reset()
        return {"accuracy": accuracy}

    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
