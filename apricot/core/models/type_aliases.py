# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import Mapping, Union, List, Callable, Tuple, Any
import numpy as np  # type: ignore


# for satisfying forward type checking
if False:  # pylint: disable=using-constant-test
    import apricot  # pylint: disable=unused-import


InternalGp = Union[
    'apricot.core.gp_internal.GpEqKernel',
    'apricot.core.gp_internal.GpM52Kernel',
    'apricot.core.gp_internal.GpM32Kernel',
    'apricot.core.gp_internal.GpRqKernel',
]
Hyperparameters = Mapping[str, np.ndarray]
# -----------------------------------------------------------------------------
MeanFunction = Union[str, int]
NoiseModel = Union[str, float]
# -----------------------------------------------------------------------------
InitData = Mapping[str, Union[int, float, np.ndarray]]
InitTypes = Union[InitData, List[str], List[int], str, int]
PyStanInitTypes = Union[List[InitData], List[str], List[int], str, int]
PyStanData = Mapping[str, Union[int, float, np.ndarray]]
# -----------------------------------------------------------------------------
LsPriorOptions = Union[str, List[str]]
LsOptObjective = Callable[[Tuple[float, float]], Tuple[float, float]]
# -----------------------------------------------------------------------------
LhsCriteria = Callable[..., float]
# -----------------------------------------------------------------------------
ObjectiveFunction = Callable[[np.ndarray], np.ndarray]
ObjectiveFunctionJac = Callable[[np.ndarray], Union[float, np.ndarray]]
# -----------------------------------------------------------------------------
Bounds = List[Tuple[float, float]]
CallbackFunction = Callable[[np.ndarray], Any]
NLML = Callable[[np.ndarray], float]
NLMLJac = Callable[[np.ndarray], Tuple[float, np.ndarray]]


