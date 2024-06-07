from functools import partial
from typing import Literal
from typing import Optional

from torchmetrics import Metric
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy
from torchmetrics.classification import F1Score
from torchmetrics.classification import FBetaScore
from torchmetrics.classification import Precision
from torchmetrics.classification import Recall
from torchmetrics.classification import Specificity
from torchmetrics.classification.auroc import AUROC
from torchmetrics.classification.average_precision import AveragePrecision

F2Score = partial(FBetaScore, beta=2.0)
F2Score.__dict__["__name__"] = "F2Score"

_DEFAULT_METRICS = [
    Accuracy,
    AUROC,
    AveragePrecision,
    F1Score,
    F2Score,
    Precision,
    Recall,
    Specificity,
]


def _remove_metrics(metrics: list, exclude: list[str]) -> list[Metric]:
    metric_mapping = {metric.__name__: metric for metric in metrics}
    return [metric_mapping[metric_name] for metric_name in metric_mapping if metric_name not in set(exclude)]


def _get_binary_metrics(
    average: Optional[Literal["macro", "micro", "sample"]] = None, prefix: str = "", exclude: Optional[list[str]] = None
) -> MetricCollection:
    raise NotImplementedError("Binary classification metrics are not implemented yet.")


def _get_multiclass_metrics(
    num_classes: int,
    average: Optional[Literal["macro", "micro", "sample"]] = None,
    prefix: str = "",
    exclude: Optional[list[str]] = None,
) -> MetricCollection:
    raise NotImplementedError("Multiclass classification metrics are not implemented yet.")


def _get_multilabel_metrics(
    num_labels: int,
    average: Optional[Literal["macro", "micro", "sample"]] = None,
    prefix: str = "",
    exclude: Optional[list[str]] = None,
) -> MetricCollection:
    if exclude is None:
        exclude = []
    metric_list = _remove_metrics(_DEFAULT_METRICS, exclude)
    if average == "macro":
        return MetricCollection(
            [metric(num_labels=num_labels, average="macro", task="multilabel") for metric in metric_list],
            prefix=prefix,
            postfix="_macro",
        )
    elif average == "micro":
        return MetricCollection(
            [metric(num_labels=num_labels, average="micro", task="multilabel") for metric in metric_list],
            prefix=prefix,
            postfix="_micro",
        )
    elif average == "sample":
        metric_list = _remove_metrics(_DEFAULT_METRICS, ["AUROC", "AveragePrecision"])
        return MetricCollection(
            [
                metric(num_labels=num_labels, average="micro", task="multilabel", multidim_average="samplewise")
                for metric in metric_list
            ],
            prefix=prefix,
            postfix="_sample",
        )
    elif average is None:
        return MetricCollection(
            [metric(num_labels=num_labels, task="multilabel", average="none") for metric in metric_list],
            prefix=prefix,
            postfix="_class",
        )
    else:
        raise ValueError(f"Invalid average {average}")


def get_classification_metric_collection(
    task: Literal["binary", "multiclass", "multilabel"],
    average: Optional[Literal["macro", "micro", "sample"]] = None,
    num_classes: Optional[int] = None,
    num_labels: Optional[int] = None,
    exclude: Optional[list[str]] = None,
    prefix: Optional[str] = None,
) -> MetricCollection:
    """
    Get a collection of classification metrics. The metrics are chosen based on the task and the average parameter.
    By default, all metrics are included. The exclude parameter can be used to exclude specific metrics. For some
    combinations of task and average, some metrics are not available and will be excluded automatically.

    Right now, only multilabel classification is implemented. The other tasks are placeholders for future
    implementations.

    The default metrics are:
        - Accuracy
        - AUROC
        - AveragePrecision
        - F1Score
        - F2Score
        - Precision
        - Recall
        - Specificity

    The following metrics are removed by default:
        - AUROC and AveragePrecision for `task="multilabel"` and `average="sample"`

    :param task: The classification task. One of "binary", "multiclass", or "multilabel".
    :param average: The averaging strategy to be used. One of "macro", "micro", "sample", or None.
    :param num_classes: The number of classes in the dataset. Required for multiclass classification.
    :param num_labels: The number of labels in the dataset. Required for multilabel classification.
    :param exclude: A list of metric names to exclude from the collection.
    :param prefix: A prefix to add to the metric names.

    :return: A MetricCollection object containing the chosen metrics.
    """
    if prefix is None:
        prefix = ""
    if task == "binary":
        assert (
            num_classes is None
        ), "num_classes should not be provided for binary classification. This hints at an error in the code."
        assert (
            num_labels is None
        ), "num_labels should not be provided for binary classification. This hints at an error in the code."
        return _get_binary_metrics(average, prefix, exclude)
    elif task == "multiclass":
        assert num_classes is not None, "num_classes should be provided for multiclass classification."
        assert (
            num_labels is None
        ), "num_labels should not be provided for multiclass classification. This hints at an error in the code."
        return _get_multiclass_metrics(num_classes, average, prefix, exclude)
    elif task == "multilabel":
        assert (
            num_classes is None
        ), "num_classes should not be provided for multilabel classification. This hints at an error in the code."
        assert num_labels is not None, "num_labels should be provided for multilabel classification."
        return _get_multilabel_metrics(num_labels, average, prefix, exclude)
    else:
        raise ValueError(f"Invalid task {task}")
