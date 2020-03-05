import os
import pytest
import numpy as np
import apricot

_MODULE = 'emulator'
_ROOTDIR = os.path.dirname(os.path.abspath(apricot.__file__))
_TESTDATAFILE = _ROOTDIR +  '/tests/testdata_{0}.npz'.format(_MODULE)
_TESTDATA = np.load(_TESTDATAFILE)

# testfunctions for hmc tests
TFS = [
    apricot.testfunctions.Higdon(),
    apricot.testfunctions.Ishigami(),
    apricot.testfunctions.Friedman(),
]

# testfunction IDs for hmc tests
NAMES = ('higdon', 'ishigami', 'friedman')

def test_hmc():
    result = {}
    for tf, func_name in zip(TFS, NAMES):
        amp_name = 'amp_{0}'.format(func_name)
        ls_name = 'ls_{0}'.format(func_name)
        x = apricot.sample_hypercube(10, tf.d, 'sobol', seed=1)
        y = tf(x)
        y = tf.normalise_with(y)
        E = apricot.fit(x, y, adapt_delta=0.9, seed=1)
        result[amp_name] = E.hyperparameters['amp']
        result[ls_name] = E.hyperparameters['ls']
        
    for r in result:
        assert(np.allclose(_TESTDATA[r], result[r]))
        
def test_methods():
    tf = apricot.testfunctions.Branin()
    x = apricot.sample_hypercube(10, tf.d, 'sobol', seed=1)
    xstar = apricot.sample_hypercube(100, tf.d, 'lhs', seed=1)
    y = tf(x)
    y = tf.normalise_with(y)
    E = apricot.fit(x, y, adapt_delta=0.9, seed=1)
    expectation_result = E(xstar)
    mu_result, sigma_result = E.marginals(xstar)
    _, Sigma_result = E.posterior(xstar)
    sobol1_result = E.sobol1(seed=1)
    ei_result = E.ei(xstar)
    px_result = E.px(xstar)
    ucb_result = E.ucb(xstar, beta=3.0)
    opt_result = E.optimise()
    xprime_result = opt_result['xprime']
    yprime_result = opt_result['val']
    assert(np.allclose(_TESTDATA['expectation_test'], expectation_result))
    assert(np.allclose(_TESTDATA['mu_test'], mu_result))
    assert(np.allclose(_TESTDATA['sigma_test'], sigma_result))
    assert(np.allclose(_TESTDATA['Sigma_test'], Sigma_result))
    assert(np.allclose(_TESTDATA['sobol1_test'], sobol1_result))
    assert(np.allclose(_TESTDATA['ei_test'], ei_result))
    assert(np.allclose(_TESTDATA['px_test'], px_result))
    assert(np.allclose(_TESTDATA['ucb_test'], ucb_result))
    assert(np.allclose(_TESTDATA['xprime_test'], xprime_result))
    assert(np.allclose(_TESTDATA['yprime_test'], yprime_result))

test_hmc()
test_methods()
