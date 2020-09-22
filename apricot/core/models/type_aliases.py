"""
Custom type aliases to be used by mypy.

-------------------------------------------------------------------------------
This file is licensed under Version 3.0 of the GNU General Public
License. See LICENSE for a text of the license.
"""
from typing import Mapping, Union, List, Callable, Tuple, Any, Literal
import numpy as np  # type: ignore


# for satisfying forward type checking
if False:  # pylint: disable=using-constant-test
    import apricot  # pylint: disable=unused-import


Zero = Literal[0]
# -----------------------------------------------------------------------------
InternalGp = Union[
    'apricot.core.gp_internal.GpEqKernel',
    'apricot.core.gp_internal.GpM52Kernel',
    'apricot.core.gp_internal.GpM32Kernel',
    'apricot.core.gp_internal.GpRqKernel',
]
Hyperparameters = Mapping[str, np.ndarray]
# -----------------------------------------------------------------------------
MeanFunction = Union[str, Zero]
NoiseModel = Union[str, float]
# -----------------------------------------------------------------------------
InitData = Mapping[str, Union[int, float, np.ndarray]]
InitTypes = Union[InitData, List[str], List[Zero], str, Zero]
PyStanInitTypes = Union[List[InitData], List[str], List[Zero], str, Zero]
PyStanData = Mapping[str, Union[int, float, np.ndarray]]
# -----------------------------------------------------------------------------
LsPriorOptions = Union[str, List[str]]
LsOptObjective = Callable[[Tuple[float, float]], Tuple[float, float]]
# -----------------------------------------------------------------------------
LhsCriteria = Callable[..., float]
# -----------------------------------------------------------------------------
# These are for emulator methods
ObjectiveFunction = Callable[[np.ndarray], np.ndarray]
ObjectiveFunctionJac = Callable[[np.ndarray], Union[float, np.ndarray]]
# -----------------------------------------------------------------------------
# For MLE and CV
Bounds = List[Tuple[float, float]]
CallbackFunction = Callable[[np.ndarray], Any]
InternalObjective = Callable[[np.ndarray], float]
InternalObjectiveJac = Callable[[np.ndarray], Tuple[float, np.ndarray]]
