import os
import pytest
import numpy as np
import apricot
from apricot.core.sampling import sample_hypercube

_MODULE = 'sampling'
_ROOTDIR = os.path.dirname(os.path.abspath(apricot.__file__))
_TESTDATAFILE = _ROOTDIR +  '/tests/testdata_{0}.npz'.format(_MODULE)
_TESTDATA = np.load(_TESTDATAFILE)

def test_sobol():
    expected = _TESTDATA['expected_sobol']
    result = sample_hypercube(32, 5, method = 'sobol', seed = 0)
    assert(np.allclose(result, expected))

def test_randomised_sobol():
    expected = _TESTDATA['expected_randomised_sobol']
    result = sample_hypercube(32, 5, method = 'randomised_sobol', seed = 0)
    assert(np.allclose(result, expected))

def test_lhs():
    expected = _TESTDATA['expected_lhs']
    result = sample_hypercube(32, 5, method = 'lhs', seed = 0)
    assert(np.allclose(result, expected))

def test_olhs():
    expected = _TESTDATA['expected_olhs']
    result = sample_hypercube(32, 5, method = 'olhs', seed = 0)
    assert(np.allclose(result, expected))

def test_mdurs():
    expected = _TESTDATA['expected_mdurs']
    result = sample_hypercube(32, 5, method = 'mdurs', seed = 0)
    assert(np.allclose(result, expected))

def test_urandom():
    expected = _TESTDATA['expected_urandom']
    result = sample_hypercube(32, 5, method = 'urandom', seed = 0)
    assert(np.allclose(result, expected))

def test_factorial():
    expected = _TESTDATA['expected_factorial']
    result = sample_hypercube(32, 5, method = 'factorial', seed = 0)
    assert(np.allclose(result, expected))

test_sobol()
test_randomised_sobol()
test_lhs()
test_olhs()
test_mdurs()
test_urandom()
test_factorial()

