(api-metrics)=
# Metrics Collection
This module contains the code for a pre-defined combination of `torchmetrics` metrics that are used in the training and
evaluation of the models. The metrics can be used for classification tasks within or independent of the ConfigILM
framework. They are a convenience feature to standardize the metrics used in the training and evaluation of the models
as well as to provide a common interface for the metrics and reduce boilerplate code. They are a simple wrapper around
the `torchmetrics` library and can be used in the same way as the metrics from the `torchmetrics` library.

:::{eval-rst}
.. automodule:: configilm.metrics
    :members:
:::
