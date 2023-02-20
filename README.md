## ConfigVLM

[![DOI](https://zenodo.org/badge/DOI/TODO)](https://doi.org/TODO)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Pipeline](https://github.com/lhackel-tub/ConfigVLM/actions/workflows/ci.yml/badge.svg)](https://github.com/lhackel-tub/ConfigVLM/actions/workflows/ci.yml)
[![Code Coverage](./coverage.svg)](./.coverage)

`ConfigVLM` is an open-source Python library for rapid iterative development of vision-language models in [`pytorch`](https://pytorch.org/). It contains implementations for easy merging of predefined and potentially pre-trained models of the [`timm`](https://github.com/rwightman/pytorch-image-models) library and [`huggingface`](https://huggingface.co/). This allows a variety of configurations of models without additional implementation effort. At the same time, the interface simplifies the exchange of components of the model and thus offers development possibilities for novel models. In addition, the package provides pre-built and throughput-optimized `pytorch dataloaders`, allowing developed models to be tested directly in various application areas such as remote sensing (RS) or common objects in context (COCO). The documentation contains installation instructions, tutorial examples, and a complete discussion of the interface to the framework.

For detailed information please visit the [publication](TODO:arXiv-Link) or [documentation](TODO:documentation).

`ConfigVLM` is released under the [Apache Software License 2.0](https://opensource.org/licenses/Apache-2.0)
