# <img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/logo_ConfigILM.png" style="font-size: 1rem; height: 2em; width: auto" alt="ConfigILM Logo"/> ConfigILM

![ConfigILM Banner](https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/ConfigILM_v1.png)

<a href="https://bifold.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/BIFOLD_Logo_farbig.png" style="font-size: 1rem; height: 2em; width: auto; margin-right: 1em" alt="BIFOLD Logo"/>
<img height="2em" hspace="10em"/>
<a href="https://www.tu.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/tu-berlin-logo-long-red.svg" style="font-size: 1rem; height: 2em; width: auto" alt="TU Berlin Logo"/>
<img height="2em" hspace="17em"/>
<a href="https://rsim.berlin/"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/RSiM_Logo_1.png" style="font-size: 1rem; height: 2em; width: auto" alt="RSiM Logo"/>
<img height="2em" hspace="17em"/>
<a href="https://eo-lab.org/de/projects/?id=12443968-ab8d-439b-8794-57d25b260406"><img src="https://raw.githubusercontent.com/wiki/lhackel-tub/ConfigILM/static/imgs/ai-cube-logo.png" style="font-size: 1rem; height: 2em; width: auto" alt="AI-Cube Logo"/>


[![Publication](https://img.shields.io/badge/Publication%20freely%20available%20on-Elsevier/SoftwareX-red.svg)](https://doi.org/10.1016/j.softx.2024.101731)


[![Release Notes](https://img.shields.io/github/release/lhackel-tub/ConfigILM)](https://github.com/lhackel-tub/ConfigILM/releases)
[![PyPI - Version](https://img.shields.io/pypi/v/configilm)](https://pypi.org/project/configilm/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/configilm)](https://pypi.org/project/configilm/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/mit-0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11562097.svg)](https://zenodo.org/records/11562097)  
[![CI Pipeline](https://github.com/lhackel-tub/ConfigILM/actions/workflows/run_tests.yml/badge.svg)](https://github.com/lhackel-tub/ConfigILM/actions/workflows/run_tests.yml)
[![CI Pipeline](https://github.com/lhackel-tub/ConfigILM/actions/workflows/build_docu.yml/badge.svg)](https://github.com/lhackel-tub/ConfigILM/actions/workflows/build_docu.yml)
[![Code Coverage](https://img.shields.io/badge/coverage%20-98%25-4c1)](./coverage.report)  
[![GitHub Star Chart](https://img.shields.io/github/stars/lhackel-tub/ConfigILM?style=social)](https://img.shields.io/github/stars/lhackel-tub/ConfigILM?style=social)
[![Open Issues](https://img.shields.io/github/issues-raw/lhackel-tub/ConfigILM)](https://github.com/lhackel-tub/ConfigILM/issues)
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

For detailed information please see its [publication](https://doi.org/10.1016/j.softx.2024.101731) 
and the [documentation](https://lhackel-tub.github.io/ConfigILM).

`ConfigILM` is released under the [MIT Software License](https://opensource.org/licenses/mit-0)

## Contributing

As an open-source project in a developing field, we are open to contributions.
They can be in the form of a new or improved feature or better documentation.

For detailed information on how to contribute, see [here](.github/CONTRIBUTING.md).


## Citation

<!-- citation-start -->
If you use this work, please cite

```bibtex
@article{hackel2024configilm,
  title={ConfigILM: A general purpose configurable library for combining image and language models for visual question answering},
  author={Hackel, Leonard and Clasen, Kai Norman and Demir, Beg{\"u}m},
  journal={SoftwareX},
  volume={26},
  pages={101731},
  year={2024},
  publisher={Elsevier}
}
```
and the used version of the software, e.g., the current version with
```bibtex
@software{hackel_2024_11562097,
  author       = {Hackel, Leonard and
                  Clasen, Kai Norman and
                  Demir, Begüm},
  title        = {ConfigILM},
  month        = jun,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.6.0},
  doi          = {10.5281/zenodo.11562097},
  url          = {https://doi.org/10.5281/zenodo.11562097}
}
```
<!-- citation-end -->

## Acknowledgement
This work is supported by the European Research Council (ERC) through the ERC-2017-STG
BigEarth Project under Grant 759764 and by the European Space Agency through the DA4DTE
(Demonstrator precursor Digital Assistant interface for Digital Twin Earth) project and
by the German Ministry for Economic Affairs and Climate Action through the AI-Cube
Project under Grant 50EE2012B. Furthermore, we gratefully acknowledge funding from the
German Federal Ministry of Education and Research under the grant BIFOLD24B.
We also thank [EO-Lab](https://eo-lab.org/en/) for giving us access to their GPUs.
