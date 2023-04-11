# ConfigVLM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7767951.svg)](https://doi.org/10.5281/zenodo.7767951)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/mit-0)
[![CI Pipeline](https://github.com/lhackel-tub/ConfigVLM/actions/workflows/ci.yml/badge.svg)](https://github.com/lhackel-tub/ConfigVLM/actions/workflows/ci.yml)
[![Code Coverage](./coverage.svg)](./.coverage)

<!-- introduction-start -->
The library `ConfigVLM` is a state-of-the-art tool for Python developers seeking to rapidly and
iteratively develop image and language models within the [`pytorch`](https://pytorch.org/) framework.
This **open-source** library provides a convenient implementation for seamlessly combining models
from two of the most popular [`pytorch`](https://pytorch.org/) libraries,
the highly regarded [`timm`](https://github.com/rwightman/pytorch-image-models) and [`huggingface`🤗](https://huggingface.co/).
With an extensive collection of nearly **1000 image** and **over 100 language models**,
with an **additional 120,000** community-uploaded models in the [`huggingface`🤗 model collection](https://huggingface.co/models),
`ConfigVLM` offers a diverse range of model combinations that require minimal implementation effort.
Its vast array of models makes it an unparalleled resource for developers seeking to create
innovative and sophisticated **image-language models** with ease.

Furthermore, `ConfigVLM` boasts a user-friendly interface that streamlines the exchange of model components,
thus providing endless possibilities for the creation of novel models.
Additionally, the package offers **pre-built and throughput-optimized**
[`pytorch dataloaders`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) and
[`lightning datamodules`](https://lightning.ai/docs/pytorch/latest/data/datamodule.html),
which enable developers to seamlessly test their models in diverse application areas, such as *Remote Sensing (RS)*.
Moreover, the comprehensive documentation of `ConfigVLM` includes installation instructions,
tutorial examples, and a detailed overview of the framework's interface, ensuring a smooth and hassle-free development experience.

<!-- introduction-end -->

![Concept of ConfigVLM](ConfigVLM-VLMType.VQA_CLASSIFICATION.png)

For detailed information please visit the [publication](TODO:arXiv-Link) or the [documentation](https://lhackel-tub.github.io/ConfigVLM).

`ConfigVLM` is released under the [MIT Software License](https://opensource.org/licenses/mit-0)


## Citation

<!-- citation-start -->
If you use this work, please cite

```bibtex
@software{lhackel-tub_2023,
  author       = {Leonard Hackel and
                  Kai Norman Clasen and
                  Begüm Demir},
  title        = {lhackel-tub/ConfigVLM: v0.1.1},
  month        = mar,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {release},
  doi          = {10.5281/zenodo.7767951},
  url          = {https://doi.org/10.5281/zenodo.7767951}
}
```
<!-- citation-end -->

## Acknowledgement
This work is supported by the European Research Council (ERC) through the ERC-2017-STG
BigEarth Project under Grant 759764 and by the European Space Agency through the DA4DTE
(Demonstrator precursor Digital Assistant interface for Digital Twin Earth) project and
by the German Ministry for Economic Affairs and Climate Action through the AI-Cube
Project under Grant 50EE2012B.

<a href="https://bifold.berlin/"><img src="BIFOLD_Logo_farbig.svg" style="font-size: 1rem; height: 2em; width: auto; margin-right: 1em" alt="BIFOLD Logo"/>
<a href="https://rsim.berlin/"><img src="RSiM_Logo_1.png" style="font-size: 1rem; height: 2em; width: auto" alt="RSiM Logo"/>