import pytest

from configilm.metrics import get_classification_metric_collection


def test_binary_metrics():
    with pytest.raises(NotImplementedError):
        get_classification_metric_collection(task="binary")


def test_binary_metrics_with_class():
    with pytest.raises(AssertionError):
        get_classification_metric_collection(task="binary", num_classes=3)


def test_binary_metrics_with_label():
    with pytest.raises(AssertionError):
        get_classification_metric_collection(task="binary", num_labels=3)


def test_multiclass_metrics():
    with pytest.raises(NotImplementedError):
        get_classification_metric_collection(task="multiclass", num_classes=3)


def test_multiclass_metrics_no_class():
    with pytest.raises(AssertionError):
        get_classification_metric_collection(task="multiclass", num_classes=None)


def test_multiclass_metrics_with_label():
    with pytest.raises(AssertionError):
        get_classification_metric_collection(task="multiclass", num_classes=3, num_labels=3)


@pytest.mark.parametrize("average", ["macro", "micro", "sample", None])
def test_multilabel_metrics(average):
    expected_len = {"macro": 8, "micro": 8, "sample": 6, None: 8}
    metric_collection = get_classification_metric_collection(task="multilabel", num_labels=3, average=average)
    assert (
        len(metric_collection) == expected_len[average]
    ), f"Expected {expected_len[average]} metrics, got {len(metric_collection)}"


def test_multilabel_metrics_invalid_average():
    with pytest.raises(ValueError):
        get_classification_metric_collection(task="multilabel", num_labels=3, average="invalid")


def test_multilabel_metrics_no_label():
    with pytest.raises(AssertionError):
        get_classification_metric_collection(task="multilabel", num_labels=None)


def test_multilabel_metrics_with_class():
    with pytest.raises(AssertionError):
        get_classification_metric_collection(task="multilabel", num_classes=3, num_labels=3)


def test_invalid_task():
    with pytest.raises(ValueError):
        get_classification_metric_collection(task="invalid")
