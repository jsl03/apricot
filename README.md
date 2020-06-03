## What is apricot?

**apricot** is a probabilistic modelling toolbox, 
built for analysing the outputs of computer codes 
using Gaussian Process (GP) regression. The focus 
of the package is on providing easy-to-build, robust
models of predictive uncertainty. 

Parameter inference for apricot's GP models is 
performed using [Stan](https://mc-stan.org/), 
interfaced with python using 
[pyStan](https://github.com/stan-dev/pystan). 
Predictive models are implemented using 
[Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
wrapped with [pybind11](https://github.com/pybind/pybind11),
and permit efficient marginalisation over the parameter 
distributions obtained with Stan's implementation of the
*No-U-Turn Sampler* (NUTS; 
[Hoffman and Gelman, 2014](www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf)).

The package also contains utilities for using the 
predictive models for tasks such as optimisation, 
adaptive sample design and sensitivity analysis.

### Okay! But what is it *for*?

Without getting into the specifics of GP regression, 
**apricot** is suited to data-sparse modelling problems
in which the data is either expensive or inconvenient 
to obtain, and there is value in understanding the 
uncertainty in the relationship between observed 
inputs and outputs.

**apricot** works best at low sample sizes 
(datasets of less than a few hundred points) 
in up to a moderate number of input dimensions 
(less than around 20).

GP regression can be performed with larger sample
sizes by introducing some approximations, though
these are currently not supported. 


## Requirements
* [SciPy](https://github.com/scipy/scipy) and [NumPy](https://github.com/numpy/numpy)

* [pyStan](https://github.com/stan-dev/pystan)

* [pybind11](https://github.com/pybind/pybind11)>=2.2

* The [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) header
  library (assumed to be in `/usr/include/eigen3`).

### Optional Extras

Both [matplotlib](https://github.com/matplotlib/matplotlib) 
and [seaborn](https://github.com/mwaskom/seaborn) are optional
dependencies which provide extra visualisation capability.


## Installation

The package currently only works on Linux. 

It can be installed using `pip3` from inside the apricot source directory:

`pip3 install --user .`

## Acknowledgements

The package was produced using research performed at Cranfield University, funded by EPSRC and Airbus UK.

### Code:

The package makes use of one or more important pieces of code from the following authors:

* apricot's `setup.py` features code directly from [pybind/python\_example](https://github.com/pybind/python_example).

* The `sobol` submodule uses code derived from [naught101/sobol\_seq](https://github.com/naught101/sobol_seq), in turn
derived from [Sobol](https://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html) by John Burkardt and Corrado Chisari.

### Important References:

* The lengthscale parameter prior distributions used inside the `priors` module closely follow considerations outlined by [Michael Betancourt](https://betanalpha.github.io/) in his three-part series ([part 1](https://betanalpha.github.io/assets/case_studies/gp_part1/part1.html), [part 2](https://betanalpha.github.io/assets/case_studies/gp_part2/part2.html), [part 3](https://betanalpha.github.io/assets/case_studies/gp_part3/part3.html)) on robust GP regression.

* **Cross Validation For Zero Mean GPs** is credited to: Sundararajan S, Keerthi SS. Predictive approaches for choosing hyperparameters in Gaussian processes. In Advances in neural information processing systems 2000 (pp. 631-637).

* **Expected Improvement** as implemented in the `Emulator` class is credited to: Mockus J, Tiesis V, Zilinskas A. The application of Bayesian methods for seeking the extremum. Towards global optimization. 1978 Dec;2(117-129):2.

* **Upper Confidence Bound** as implemented in the `Emulator` class is credited to: Srinivas N, Krause A, Kakade SM, Seeger M. Gaussian process optimization in the bandit setting: No regret and experimental design. arXiv preprint arXiv:0912.3995. 2009 Dec 21.

* **Input Warping For Non-Stationary GP Regression** in credited to: Snoek J, Swersky K, Zemel R, Adams R. Input warping for Bayesian optimization of non-stationary functions. InInternational Conference on Machine Learning 2014 Jan 27 (pp. 1674-1682).

## Similar and Related Packages

* [GPy](https://github.com/SheffieldML/GPy) is a feature-rich package for performing GP regression in python.
* [PyMC3](https://github.com/pymc-devs/pymc3) provides general probabilistic modelling capabilities in python, including GP regression.
* [pyStan](https://github.com/stan-dev/pystan) is the python interface to the probabilistic programming language [Stan](https://mc-stan.org/).

## License

GPL v3. See LICENSE for a copy of the license. 
