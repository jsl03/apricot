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
adaptive sample design, and sensitivity analysis.

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
(less than around 20). Please see the limitations section below for more information.

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

The package currently only works "as is" on Linux. It can be installed using `pip` from inside the apricot source directory:

`pip install --user .`

Please note that the package assumes the `Eigen` headers are located in `usr/include/eigen3`. Ability to change this location is planned for a future release. Those needing to change the default `Eigen` header location should modify the `ext_modules` variable inside `setup.py` to include the proper location of the `Eigen` headers.

### A (Very) Brief Troubleshooting Guide For Prospective Windows or OSX Users

I anticipate you will need to make the following changes:

* Modify `setup.py` so that the C++ files will compile correctly.

* Modify the paths inside  `apricot/core/models/model_cache` to support the correct path formats for your operating system.

Please note that the package has **not** be tested on Windows or OSX at all.

## Limitations

#### Why less than a few hundred points?

The "model fit" stage of "vanilla" GP regression, as implemented by `apricot`, involves 
a pairwise calculation between each observed data point/sample. So, for a sample size of `N`, this produces a symmetrical matrix of size `(N,N)`.
We then need to invert this matrix (or calculate it's Cholesky factor, which is more computationally convenient), which (together with the previous step)
implies a time complexity of `O(N^3)`: This becomes prohibitive very quickly as `N` gets larger (without employing some special techniques, at least).

Additionally, for "large" `N` (in scare quotes because by modern standards, "large" is *very* relative), 
storage of these matrices (which, being square, are of size `N^2`) can also be an issue.

In practical terms, this means how slow the fit procedure is scales cubicly with the number of samples, 
and things can become unreasonably slow quite fast as a result. 
Exactly when this "slowness" becomes prohibitive depends on the available hardware and the application context, of course:
a model fit time of an hour or so might be acceptable if you only intend on doing the analysis once, while
a fit time of even a few minutes might be too slow if the method is part of a larger procedure (such as, for example, a Bayesian optimisation routine).

#### Why less than about 20 input dimensions?

Scaling with regards to the number of input *dimensions*, `D`, is a little more nuanced.

[Michael Betancourt does a far better job of explaining this "curse of dimensionality" in the context of GP regression than I could](https://betanalpha.github.io/assets/case_studies/gp_part3/part3.html#6_the_inevitable_curse_of_dimensionality),  
but the gist of it is that GP regression works by assessing some measure of "distance" (in scare quotes because this need not be a distance in the conventional sense) between points, and then determining how similar two function values ought to be based on this (with points that are "close together" or "similar" typically having similar values). 

By adding more dimensions (informally, more axes of comparison that data points can differ across), we necessarily "spread out" the data more, and hence need more points to provide an equivalent amount of coverage in terms of the distance mentioned in the previous paragraph. While this isn't *quite* the full story (it is a *little* more complicated than this, and relates to how the GP model is structured, too), this limitation interacts with the first issue (scaling with `N`) in that we start to require more sample points than is computationally sensible if `D` becomes large.

#### I want to use GP Regression for precisely those problems!

You're not alone! This is an area of active research. For currently existing solutions I suggest taking a look at the low-rank (or "sparse") and/or "special structure" GP models in either [GPyTorch](https://gpytorch.ai/) or [GPflow](https://github.com/GPflow/GPflow).

For a good introduction to the literature (and so you can understand the options presented by the above two packages), 
I would suggest starting with either of these two papers and following some of the references, depending on the specifics of your problem.

* [Understanding Probabilistic Sparse Gaussian Process Approximations](https://papers.nips.cc/paper/6477-understanding-probabilistic-sparse-gaussian-process-approximations.pdf), by Bauer, van der Wilk, and Rasmussen. 

* [Thoughts on Massively Scalable Gaussian Processes](https://arxiv.org/pdf/1511.01870.pdf), by Wilson, Dann, and Nickisch. 

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

* **Input Warping For Non-Stationary GP Regression** in credited to: Snoek J, Swersky K, Zemel R, Adams R. Input warping for Bayesian optimization of non-stationary functions. International Conference on Machine Learning 2014 Jan 27 (pp. 1674-1682).

## Similar and Related Packages

* [GPy](https://github.com/SheffieldML/GPy) is a feature-rich package for performing GP regression in python.
* [PyMC3](https://github.com/pymc-devs/pymc3) provides general probabilistic modelling capabilities in python, including GP regression.
* [pyStan](https://github.com/stan-dev/pystan) is the python interface to the probabilistic programming language [Stan](https://mc-stan.org/).
* [GPyTorch](https://gpytorch.ai/) provides "a highly efficient and modular implementation" of GP regression, implemented in [PyTorch](https://pytorch.org/).
* [GPflow](https://github.com/GPflow/GPflow) offers similar features to GPyTorch, using [TensorFlow](https://tensorflow.org/) instead.


## License

GPL v3. See LICENSE for a copy of the license. 
