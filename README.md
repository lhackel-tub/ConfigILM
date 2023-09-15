# ConfigILM
![ConfigILM Logo](https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/ConfigILM_v1.png)

<a href="https://bifold.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/BIFOLD_Logo_farbig.png" style="font-size: 1rem; height: 2em; width: auto; margin-right: 1em" alt="BIFOLD Logo"/>
<img height="2em" hspace="10em"/>
<a href="https://www.tu.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/tu-berlin-logo-long-red.svg" style="font-size: 1rem; height: 2em; width: auto" alt="TU Berlin Logo"/>
<img height="2em" hspace="17em"/>
<a href="https://rsim.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/RSiM_Logo_1.png" style="font-size: 1rem; height: 2em; width: auto" alt="RSiM Logo"/>


[![Release Notes](https://img.shields.io/github/release/lhackel-tub/ConfigILM)](https://github.com/lhackel-tub/ConfigILM/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8348912.svg)](https://doi.org/10.5281/zenodo.8348912)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/mit-0)
[![CI Pipeline](https://github.com/lhackel-tub/ConfigILM/actions/workflows/run_tests.yml/badge.svg)](https://github.com/lhackel-tub/ConfigILM/actions/workflows/run_tests.yml)
[![CI Pipeline](https://github.com/lhackel-tub/ConfigILM/actions/workflows/build_docu.yml/badge.svg)](https://github.com/lhackel-tub/ConfigILM/actions/workflows/build_docu.yml)
[![Code Coverage](./coverage.svg)](./.coverage)
[![GitHub Star Chart](https://img.shields.io/github/stars/lhackel-tub/ConfigILM?style=social)](https://img.shields.io/github/stars/lhackel-tub/ConfigILM?style=social)
[![Open Issues](https://img.shields.io/github/issues-raw/lhackel-tub/ConfigILM)](https://github.com/lhackel-tub/ConfigILM/issues)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/configilm)](https://pypi.org/project/configilm/)
[![PyPI - Version](https://img.shields.io/pypi/v/configilm)](https://pypi.org/project/configilm/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/configilm)](https://pypi.org/project/configilm/)


<!-- introduction-start -->
The library `ConfigILM` is a state-of-the-art tool for Python developers seeking to rapidly and
iteratively develop image and language models within the [`pytorch`](https://pytorch.org/) framework.
This **open-source** library provides a convenient implementation for seamlessly combining models
from two of the most popular [`pytorch`](https://pytorch.org/) libraries,
the highly regarded [`timm`](https://github.com/rwightman/pytorch-image-models) and [`huggingface`🤗](https://huggingface.co/).
With an extensive collection of nearly **1000 image** and **over 100 language models**,
with an **additional 120,000** community-uploaded models in the [`huggingface`🤗 model collection](https://huggingface.co/models),
`ConfigILM` offers a diverse range of model combinations that require minimal implementation effort.
Its vast array of models makes it an unparalleled resource for developers seeking to create
innovative and sophisticated **image-language models** with ease.

Furthermore, `ConfigILM` boasts a user-friendly interface that streamlines the exchange of model components,
thus providing endless possibilities for the creation of novel models.
Additionally, the package offers **pre-built and throughput-optimized**
[`pytorch dataloaders`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) and
[`lightning datamodules`](https://lightning.ai/docs/pytorch/latest/data/datamodule.html),
which enable developers to seamlessly test their models in diverse application areas, such as *Remote Sensing (RS)*.
Moreover, the comprehensive documentation of `ConfigILM` includes installation instructions,
tutorial examples, and a detailed overview of the framework's interface, ensuring a smooth and hassle-free development experience.

<!-- introduction-end -->

![Concept of ConfigILM](https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/ConfigILM-ILMType.VQA_CLASSIFICATION.png)

For detailed information please visit the [publication](TODO:arXiv-Link) or the [documentation](https://lhackel-tub.github.io/ConfigILM).

`ConfigILM` is released under the [MIT Software License](https://opensource.org/licenses/mit-0)

## Contributing

As an open-source project in a developing field, we are open to contributions, whether it be in the form of a new or improved feature or better documentation.

For detailed information on how to contribute, see [here](.github/CONTRIBUTING.md).


## Citation

<!-- citation-start -->
If you use this work, please cite

```bibtex
@software{lhackel_tub_2023,
  author       = {Hackel, Leonard and
                  Clasen, Kai Norman and
                  Demir, Begüm},
  title        = {ConfigILM},
  month        = sep,
  year         = 2023,
  note         = {{If you use this software, please cite it using the
                   metadata from this file.}},
  publisher    = {Zenodo},
  version      = {v0.4.6},
  doi          = {10.5281/zenodo.8348912},
  url          = {https://doi.org/10.5281/zenodo.8348912}
}
```
<!-- citation-end -->

## Acknowledgement
This work is supported by the European Research Council (ERC) through the ERC-2017-STG
BigEarth Project under Grant 759764 and by the European Space Agency through the DA4DTE
(Demonstrator precursor Digital Assistant interface for Digital Twin Earth) project and
by the German Ministry for Economic Affairs and Climate Action through the AI-Cube
Project under Grant 50EE2012B.
