## What is apricot?

**apricot** is a probabilistic modelling toolbox for analysing the outputs of computer codes using Gaussian Process (GP) regression.

Parameter inference for apricot's GP models is performed using [Stan](https://mc-stan.org/) interfaced through [pyStan](https://github.com/stan-dev/pystan), and the resulting predictive models are implemented using [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) wrapped with [pybind11](https://github.com/pybind/pybind11).

**apricot** assembles relatively simple GP regression models in Stan code automatically from generic program "blocks", such that Stan's implementation of the *No-U-Turn Sampler* (NUTS; [Hoffman and Gelman, 2014](www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf)) can be used to perform efficient sampling of the model parameters, given data. Samples of the model parameters can then be used directly by **apricot**'s compiled predictive models to perform tasks including surrogate modelling/emulation, optimisation, and sensitivity analysis.

## Requirements
* [SciPy](https://github.com/scipy/scipy) and [NumPy](https://github.com/numpy/numpy)

* [pyStan](https://github.com/stan-dev/pystan)

* [pybind](https://github.com/pybind/pybind11)>=2.2

* The [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) header library.

Both [matplotlib](https://github.com/matplotlib/matplotlib) and [seaborn](https://github.com/mwaskom/seaborn) are optional dependencies which provide extra visualisation capability.


## Installation

**apricot** has currently only been tested on Linux. 

It can be installed using `pip3` from inside the apricot source directory:

`pip3 install --user .`

## Acknowledgements

This package makes use of one or more pieces of code from the following authors:

* **apricot**'s `setup.py` features code directly from [pybind/python\_example](https://github.com/pybind/python_example).

* The sobol sampling submodule uses code derived from [naught101/sobol\_seq](https://github.com/naught101/sobol_seq).

* The lengthscale parameter prior distributions used inside **apricot** closely follow considerations outlined by [Michael Betancourt](https://betanalpha.github.io/) in his three-part series ([part 1](https://betanalpha.github.io/assets/case_studies/gp_part1/part1.html), [part 2](https://betanalpha.github.io/assets/case_studies/gp_part2/part2.html), [part 3](https://betanalpha.github.io/assets/case_studies/gp_part3/part3.html)) on robust GP regression.

