import numpy as np  # type: ignore
from apricot.core.utils import mad


_twopi = np.pi * 2.0


# TODO there is a math function for this
def _to_rad(deg):
    return deg * (np.pi / 180.0)


def _cosd(x):
    return np.cos(_to_rad(x))


class TestFunction:
    """ Test Function base class

    Instances of specific testfunctions should derive from this class.

    Attributes
    ----------
    d : int
        Test function input dimension.
    bounds : list of tuple
        Length d list of bounds described by (lower, upper) tuples.
    f : callable
        Function accepting (n,d) array of points with bounds [lower_i, upper_i]
        i = 1 ... d.
    noise : float
        Standard deviation of additive Gaussian noise to apply to function
        responses. Set to None to make the function noiseless.
    mean : float
        Mean used for automatic normalisation. None if instance is not currently
        normalised.
    mad : float
        Mean Absolute Deviation (MAD) used for automatic normalisation. None if
        instance is not currently normalised. The MAD is a robust measure of
        dispersion.

    Methods
    -------
    set_normalisation
        Initialise automatic normalisation using the mean and median absolute
        deviation of a supplied data sample.
    reset_normalisation
        Reset automatic normalisation.
    __call__
        Query the test function. Accepts inputs on [0,1]^self.d before
        rescaling them into [upper_i, lower_i] for i = 1...self.d. If a noise
        standard deviation is supplied, additive Gaussian noise with the
        requested standard deviation will be added to the responses. If
        automatic normalisation is enabled, the noisy responses will then have
        self.mean subtracted from them and be divided by self.mad.

    Private Attributes
    ------------------
    _normalised : bool
        Indicates whether the function instances currently features automatic
        normalisation or not.
    _lower : ndarray
        (d,) array of input lower bounds
    _upper : ndarray
        (d,) array of input upper bounds

    Private Methods
    ---------------
    _format_input
        Internal method. Formats inputs to __call__ to be correctly scaled and
        shaped.
    _call_with_noise
        Internal method. Query the function with additive Gaussian noise that
        has standard deviation equal to self.noise. Any rescaling is applied
        after calling this function.
    _call_no_noise
        Internal method. Query the function without noise. Any rescaling is
        applied after calling this function.
    """

    def __init__(self, f, bounds, noise=None):
        self.bounds = bounds
        self.f = f
        self.noise = noise
        self._normalised = False

    @property
    def d(self):
        return len(self.bounds)

    @property
    def _lower(self):
        """ Return lower bounds for each input dimension"""
        return np.array([a for a, _ in self.bounds])

    @property
    def _upper(self):
        """ Return upper bounds for each input dimension"""
        return np.array([b for _, b in self.bounds])

    def set_normalisation(self, y):
        """
        Set normalisation for the test function used for subsequent queries
        based on the mean and standard deviation of the provided sample, and
        return the provided values scaled by the stored normalisation
        constants.

        Parameters
        ----------
        y : data
            (n,) array

        Returns
        -------
        y_normalised : normalised data
            (n,) array
        """
        self._normalised = True
        self.mean = np.mean(y, dtype=np.float64)
        self.mad = mad(y)
        return (y - self.mean) / self.mad

    def reset_normalisation(self):
        """ Reset normalisation on the test function instance"""
        self._normalised = False
        self.mean = None
        self.mad = None

    def __call__(self, x):
        """ Query the test function"""

        if self.noise:
            y = self._call_with_noise(x)
        else:
            y = self._call_no_noise(x)

        if self._normalised:
            return (y - self.mean)/self.mad
        else:
            return y

    def _format_input(self, x):
        """ Format test function inputs

        Ensures an array, x, of points on [0,1], is of the correct shape,
        before rescaling the array such that the dimensionwise bounds are those
        described by self._lower and self._upper.

        Parameters
        ----------
        x : ndarray
            (n, self.d) array of points on [0,1]^d

        Returns
        -------
        xbar : ndarray array
            (n, self.d) array of points on [lower_i, upper_i], i=1...d.

        Notes
        -----
        Will reshape 1D arrays to be of shape (n,1) automatically
        """
        x = np.atleast_1d(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        xbar = (x * (self._upper - self._lower)) + self._lower
        return xbar

    def _call_with_noise(self, x):
        """ Query the test function with added Gaussian noise"""
        y = self.f(self._format_input(x))
        noise = np.random.normal(loc=0, scale=self.noise, size=y.shape)
        return (y + noise).ravel()

    def _call_no_noise(self, x):
        """ Query the test function"""
        y = self.f(self._format_input(x))
        return y.ravel()

# -----------------------------------------------------------------------------
# Wing weight function
# -----------------------------------------------------------------------------


class Wing_Weight(TestFunction):
    """ Forrester, A., Sobester, A., & Keane, A. (2008)

    Notes
    -----
    https://www.sfu.ca/~ssurjano/wingweight.html
    """

    def __init__(self, noise=None):
        self.bounds = _bounds_wing_weight
        self.f = _f_wing_weight
        self.noise = noise
        self._normalised = False

       
_bounds_wing_weight = [
    (150, 200),
    (220, 300),
    (6, 10),
    (-10, 10),
    (16, 45),
    (0.5, 1),
    (0.08, 0.18),
    (2.5, 6),
    (1700, 2500),
    (0.025, 0.08)]


def _f_wing_weight(x):
    tmp1 = 0.036*x[:, 0]**0.758
    tmp2 = x[:, 1]**0.0035
    tmp3 = (x[:, 2] / (_cosd(x[:, 3])**2))**0.6
    tmp4 = x[:, 4]**0.006 * x[:, 5]**0.04
    tmp5 = (100 * x[:, 6] / _cosd(x[:, 3]))**(-0.3)
    tmp6 = (x[:, 7]*x[:, 8])**0.49
    return tmp1 * tmp2 * tmp3 * tmp4 * tmp5 * tmp6 + (x[:, 0] * x[:, 9])

# -----------------------------------------------------------------------------
# Circuit function
# -----------------------------------------------------------------------------


class Circuit(TestFunction):
    """ Ben-Ari, E. N., & Steinberg, D. M. (2007)

    Notes
    -----
    https://www.sfu.ca/~ssurjano/otlcircuit.html
    """

    def __init__(self, noise=None):
        self.bounds = _bounds_circuit
        self.f = _f_circuit
        self.noise = noise
        self._normalised = False

       
_bounds_circuit = [
    (50, 150),
    (25, 70),
    (0.5, 3),
    (1.2, 2.5),
    (0.25, 1.2),
    (50, 300),
]


def _f_circuit(x):
    vb1 = (12.0 * x[:, 1]) / (x[:, 0] + x[:, 1])
    tmp1 = ((vb1 + 0.74) * x[:, 5] * (x[:, 4] + 9.0)) / ((x[:, 5] * (x[:, 4] + 9)) + x[:, 2])
    tmp2 = (11.35 * x[:, 2]) / ((x[:, 5]*(x[:, 4] + 9.0)) + x[:,2])
    tmp3 = (0.74 * x[:, 2] * x[:, 5] * (x[:, 4] + 9.0)) / (((x[:, 5] * (x[:, 4] + 9)) + x[:, 2]) * x[:, 3])
    return tmp1 + tmp2 + tmp3

# -----------------------------------------------------------------------------
# Branin function
# -----------------------------------------------------------------------------


class Branin(TestFunction):
    """ Dixon, L. C. W., & Szego, G. P. (1978).

    Notes
    -----
    See: https://www.sfu.ca/~ssurjano/branin.html
    """

    def __init__(self, noise=None):
        self.bounds = _bounds_branin
        self.f = _f_branin
        self.noise = noise
        self._normalised = False


_bounds_branin = [
    (-5, 10),
    (0, 15),
]


def _f_branin(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    tmp1 = (-1.275*(x[:, 0]**2)/(np.pi**2) + 5*x[:, 0]/np.pi + x[:, 1] - 6)**2
    tmp2 = (10 - 5 / (np.pi*4)) * np.cos(x[:, 0]) + 10
    return tmp1 + tmp2

# -----------------------------------------------------------------------------
# Cosines function
# -----------------------------------------------------------------------------


class Cosines(TestFunction):
    """ L. Breiman, and A. Cutler (1993).

    Accepts an additional input parameter 'd' which determines the number of
    input dimensions.

    Notes
    -----
    Sometimes "mixture of cosines" or "cosine mixture".
    """

    def __init__(self, d=2, noise=None):
        self.bounds = [(-1, 1) for _ in range(d)]
        self.f = _f_cosines
        self.noise = noise
        self._normalised = False


def _f_cosines(x):
    tmp1 = -0.1 * np.sum(np.cos(5.0 * np.pi * x), axis=1)
    tmp2 = np.sum(x**2, axis=1)
    return -(tmp1 - tmp2)

# -----------------------------------------------------------------------------
# Borehole function 
# -----------------------------------------------------------------------------

class Borehole(TestFunction):
    """ An, J., & Owen, A. (2001)
    
    Notes
    -----
    See: https://www.sfu.ca/~ssurjano/borehole.html
    """
    def __init__(self, noise=None):
        self.bounds = _bounds_borehole
        self.f = _f_borehole
        self.noise = noise
        self._normalised = False

_bounds_borehole = [
    (0.05, 0.15),
    (100, 50000),
    (63070, 115600),
    (990, 1110),
    (63.1, 116),
    (700, 820),
    (1120, 1680),
    (9855, 12045)]

def _f_borehole(x):
    tmp1 = _twopi*x[:,2]*(x[:,3]-x[:,5])
    tmp2 = np.log(x[:,1]/x[:,0])
    tmp3 = 1.0 + ((2.0 * x[:,6] * x[:,2]) / (np.log(x[:,1]/x[:,0])*(x[:,0]**2)*x[:,7])) + (x[:,2] / x[:,4])
    return tmp1 / (tmp2 * tmp3)

# -----------------------------------------------------------------------------
# Robot arm function 
# -----------------------------------------------------------------------------

class Robot_Arm(TestFunction):
    """ An, J., & Owen, A. (2001)
    
    Notes
    -----
    See: https://www.sfu.ca/~ssurjano/robot.html
    """
    def __init__(self, noise=None):
        self.bounds = _bounds_arm
        self.f = _f_arm
        self.noise = noise
        self._normalised = False

_bounds_arm = [
    (0, _twopi),
    (0, _twopi),
    (0, _twopi),
    (0, _twopi),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
]
def _f_arm(x):
    n = x.shape[0]
    u = np.sum(x[:,4:] * np.broadcast_to(np.cos(np.sum(x[:,0:4], axis=1)),(4, n)).T, axis=1)
    v = np.sum(x[:,4:] * np.broadcast_to(np.sin(np.sum(x[:,0:4], axis=1)),(4, n)).T, axis=1)
    return (u**2. + v**2.)**0.5

# -----------------------------------------------------------------------------
# Piston function 
# -----------------------------------------------------------------------------

class Piston(TestFunction):
    """ Kenett, R., & Zacks, S. (1998)
    
    Notes
    -----
    See: https://www.sfu.ca/~ssurjano/piston.html
    """
    def __init__(self, noise=None):
        self.bounds = _bounds_piston
        self.f = _f_piston
        self.noise = noise
        self._normalised = False

_bounds_piston = [
    (30, 60),
    (0.005, 0.020),
    (0.002, 0.010),
    (1000, 5000),
    (90000, 110000),
    (290, 296),
    (340, 360),
]
def _f_piston(x):
    A = (x[:,4]*x[:,1]) + 19.62*x[:,0] - ((x[:,3]*x[:,2])/x[:,1])
    V = (x[:,1]/2.0*x[:,3])*(np.sqrt(A**2+4.0*x[:,3]*((x[:,4]*x[:,2])/x[:,6])*x[:,5])-A)
    tmp = x[:,3] + (x[:,1]**2 * ((x[:,4] * x[:,2])/x[:,6]) * (x[:,5]/V**2))
    return _twopi * np.sqrt(x[:,0] / tmp)

# -----------------------------------------------------------------------------
# Sphere function 
# -----------------------------------------------------------------------------

class Sphere(TestFunction):
    """ n-dimensional sphere function

    Accepts an additional input parameter 'd' which determines the number of
    input dimensions.

    """
    def __init__(self, d=2, noise=None):
        self.bounds = [(-1, 1) for _ in range(d)]
        self.f = _f_sphere
        self.noise = noise
        self._normalised = False

def _f_sphere(x):
    return np.sum(x**2, axis=1)

# -----------------------------------------------------------------------------
# Friedman function 
# -----------------------------------------------------------------------------

class Friedman(TestFunction):
    """ Friedman, J. H., Grosse, E., & Stuetzle, W. (1983)
    
    Notes
    -----
    See: https://www.sfu.ca/~ssurjano/fried.html
    """
    def __init__(self, noise=None):
        self.bounds = _bounds_friedman
        self.f = _f_friedman
        self.noise = noise
        self._normalised = False

_bounds_friedman = [(0, 1) for _ in range(5)]

def _f_friedman(x):
    a =10*np.sin(np.pi*x[:,0]*x[:,1])
    b = 20*(x[:,2] - 0.5)**2
    c = 10*x[:,3]
    d = 5*x[:,4]
    return a + b + c + d

# -----------------------------------------------------------------------------
# Higdon function 
# -----------------------------------------------------------------------------

class Higdon(TestFunction):
    """ Higdon, D. (2002)
    
    Notes
    -----
    See: https://www.sfu.ca/~ssurjano/hig02.html
    """
    def __init__(self, noise=None):
        self.bounds = _bounds_higdon
        self.f = _f_higdon
        self.noise = noise
        self._normalised = False

_bounds_higdon = [(0, 10)]

def _f_higdon(x):
    x = x.ravel()
    return np.sin((2.0*np.pi*x)/10) + 0.2*np.sin((2.0*np.pi*x)/2.5)

# -----------------------------------------------------------------------------
# Gramacy-Lee function 
# -----------------------------------------------------------------------------

class Gramacy_Lee(TestFunction):
    """ Gramacy, R. B., & Lee, H. K. (2009)

    This is Gramacy & Lee's 2009 function.
    
    Notes
    -----
    See: https://www.sfu.ca/~ssurjano/grlee09.html
    """
    def __init__(self, noise=None):
        self.bounds = _bounds_gramacy_lee
        self.f = _f_gramacy_lee
        self.noise = noise
        self._normalised = False

_bounds_gramacy_lee = [(0, 1) for _ in range(4)]

def _f_gramacy_lee(x):
    inner = (0.9*(x[:,0]+0.48))**10
    return np.exp(np.sin(inner))+(x[:,1] * x[:,2])+x[:,3]

# -----------------------------------------------------------------------------
# 3-hump camel function 
# -----------------------------------------------------------------------------

class Camel3(TestFunction):
    """ 3-Hump Camel Test Function
    
    Notes
    -----
    See: https://www.sfu.ca/~ssurjano/camel3.html
    """
    def __init__(self, noise=None):
        self.bounds = _bounds_camel3
        self.f = _f_camel3
        self.noise = noise
        self._normalised = False

_bounds_camel3 = [(-5, 5) for _ in range(2)]

def _f_camel3(x):
    t1 = 2.0*x[:,0]**2
    t2 = -1.05*x[:,0]**4.
    t3 = ((x[:,0]**6.)/6.)
    t4 = x[:,0]*x[:,1]
    t5 = x[:,1]**2.
    return t1 + t2 + t3 + t4 + t5

# -----------------------------------------------------------------------------
# 6-hump camel function 
# -----------------------------------------------------------------------------

class Camel6(TestFunction):
    """ 6-Hump Camel Test Function
    
    Notes
    -----
    See: https://www.sfu.ca/~ssurjano/camel6.html
    """
    def __init__(self, noise=None):
        self.bounds = _bounds_camel6
        self.f = _f_camel6
        self.noise = noise
        self._normalised = False

_bounds_camel6 = [(-3, 3), (-2, 2)]

def _f_camel6(x):
    t1 = (4. - 2.1*x[:,0]**2. + (x[:,0]**4.)/3.) * x[:,0]**2
    t2 = x[:,0] * x[:,1]
    t3 = (-4. + 4.0 * x[:,1]**2) * x[:,1]**2
    return t1 + t2 + t3

# -----------------------------------------------------------------------------
# Ishigami function 
# -----------------------------------------------------------------------------


class Ishigami(TestFunction):
    """ Ishigami, T., & Homma, T. (1990)

    Notes
    -----
    Accepts two additional parameters a and b controlling the behaviour of the
    test function.

    See: https://www.sfu.ca/~ssurjano/ishigami.html
    """

    def __init__(self, a=0.7, b=0.1, noise=None):
        self.bounds = _bounds_ishigami
        self.f = _make_ishigami(a, b)
        self.noise = noise
        self._normalised = False


_bounds_ishigami = [(-np.pi, np.pi) for _ in range(3)]


def _make_ishigami(a, b):
    """ Closure to return Ishigami test function with parameters a and b"""
    def _f_ishigami(x):
        return (
            np.sin(x[:, 0]) +
            a*(np.sin(x[:, 1])**2)+
            b*x[:, 2]**4 * np.sin(x[:, 0])
        )
    return _f_ishigami

# ------------------------------------------------------------------------------
# Franke's function
# ------------------------------------------------------------------------------


class Franke(TestFunction):
    """ Franke, R (1979)

    Notes
    -----
    See: https://www.sfu.ca/~ssurjano/franke2d.html

    """

    def __init__(self, noise=None):
        self.bounds = [(0, 1)] * 2
        self.f = _f_franke
        self.noise = noise
        self._normalised = False


def _f_franke(x):
    return (
        0.75 * np.exp(
            - (((9 * x[:, 0] - 2)**2) / 4)
            - (((9 * x[:, 1] - 2)**2) / 4)
        ) +
        0.75 * np.exp(
            - (((9 * x[:, 0] + 1)**2) / 49)
            - ((9 * x[:, 1] + 1) / 10)
        ) +
        0.5 * np.exp(
            - (((9 * x[:, 0] - 7)**2) / 4)
            - (((9 * x[:, 1] - 3)**2) / 4)
        ) -
        0.2 * np.exp(-(9 * x[:, 0] - 4)**2 - (9 * x[:, 1] - 7)**2)
    )
