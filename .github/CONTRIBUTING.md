(contributing)=
# Contributing
This project is a community effort, and everyone is welcome to contribute!

Thank you for taking a look at how you can contribute to the project! 💚
There are many ways to help this framework and documentation grow.
In the following, we will list a couple of ways on how you can contribute to this project, or more generally, to the broader community 😎.

## Give feedback
The easiest way to improve the framework or documentation is to highlight mistakes or areas that need further refinement.

If you would like to provide feedback and/or discuss something or if you have a feature request, feel free to open an _issue_ on the {{issues}} page.


## Increasing visibility
- Tell others about this framework!
- [Star 🌟 the project on GitHub](https://github.com/lhackel-tub/ConfigILM/)

## Directly update source code or notebooks

[Poetry](https://python-poetry.org/) is used to manage all [Python](https://www.python.org/) dependencies.

To set up the development environment
1. [Download and install Poetry](https://python-poetry.org/). Make sure to use a valid python version in the environment that you install Poetry in. Check the README for details.
2. Clone this repository with `git clone https://github.com/lhackel-tub/ConfigILM/`
3. Change to the downloaded directory with `cd ConfigILM`
4. Replicate the development environment (with all additional dependencies) with `poetry install --with dev -E full` (this may take some time depending on your download speed)
5. Start hacking 😁

If you create a [PR](https://docs.github.com/en/get-started/quickstart/hello-world#opening-a-pull-request), an automatic test suite will check if the framework still passes its requirements and the documentation can be successfully regenerated.
After all tests pass, we will try our best to get back to you as quickly as possible!
