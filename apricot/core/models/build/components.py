# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------
from typing import Union, List, Optional, Any
from apricot.core import utils
from apricot.core.models.build import filenames


ModelCode = Optional[Union[List[str], str]]


# TODO some options are deprecated (is args still used?)
class StanModelPart:
    """ Base class for the Stan model "parts" which are combined to form a
    pyStan model interface.
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments
            self,
            name: Optional[str] = None,
            functions: ModelCode = None,
            data: ModelCode = None,
            transformed_data: ModelCode = None,
            parameters: ModelCode = None,
            transformed_parameters: ModelCode = None,
            model: ModelCode = None,
            args: ModelCode = None,
            to_sample: ModelCode = None,
            data_priors: ModelCode = None,
    ):

        self._name = name
        functions = none_to_list(functions)
        data = none_to_list(data)
        transformed_data = none_to_list(transformed_data)
        paramaters = none_to_list(parameters)
        transformed_parameters = none_to_list(transformed_parameters)
        model = none_to_list(model)
        args = none_to_list(args)
        to_sample = none_to_list(to_sample)
        data_priors = none_to_list(data_priors)

        # Stan code blocks
        self._functions = utils.join_lines(functions)
        self._data = utils.join_lines(data)
        self._transdata = utils.join_lines(transformed_data)
        self._params = utils.join_lines(parameters)
        self._transparams = utils.join_lines(transformed_parameters)
        self._model = utils.join_lines(model)

        # misc
        self._args = utils.to_list(args)
        self._to_sample = utils.to_list(to_sample)
        self._data_priors = utils.to_list(data_priors)

    @property
    def filename_component(self):
        """ This part's contribution to the unique model filname."""
        return self._name

    @property
    def functions(self):
        """ Contribution to Stan model "functions" block."""
        return self._functions

    @property
    def data(self):
        """ Contribution to Stan model "data" block."""
        return self._data

    @property
    def transdata(self):
        """ Contribution to Stan model "transformed data" block."""
        return self._transdata

    @property
    def params(self):
        """ Contribution to Stan model "parameters" block."""
        return self._params

    @property
    def transparams(self):
        """ Contribution to Stan model "transformed parameters" block."""
        return self._transparams

    @property
    def model(self):
        """Contribution to Stan model "model" block."""
        return self._model

    @property
    def args(self):
        """ Contribution to the arguments required to run the (fit) model. For
        example, a linear mean requires beta to be present when computing the
        posterior (predictive) distribution.
        """
        return self._args

    @property
    def to_sample(self):
        """ Contribution to the parameters to be sampled by the model. Passed
        to pystan as the "pars" variable when invoking pystan.StanModel
        methods.
        """
        return self._to_sample

    @property
    def data_priors(self):
        """ Contribution to the entries required in the "data" dictionary
        required by the model.
        """
        return self._data_priors

    def __iter__(self):
        # calling iter on the class accesses a generator yielding the model
        # code blocks, for example:
        # >>> for block in model_component:
        # >>>     <build the model>
        block_gen = (c for c in (
            self.functions,
            self.data,
            self.transdata,
            self.params,
            self.transparams,
            self.model))
        return block_gen


class StanModelKernel(StanModelPart):
    """ Stan model kernels. """
    # pylint: disable=too-many-instance-attributes
    def __init__(  # pylint: disable=too-many-arguments, too-many-locals
            self,
            name: Optional[str] = None,
            functions: ModelCode = None,
            data: ModelCode = None,
            transformed_data: ModelCode = None,
            parameters: ModelCode = None,
            transformed_parameters: ModelCode = None,
            model: ModelCode = None,
            kernel_signature: Optional[str] = None,
            args: ModelCode = None,
            to_sample: ModelCode = None,
            data_priors: ModelCode = None,
    ):

        self._name = name

        # Stan code blocks
        self._functions = utils.join_lines(functions)
        self._data = utils.join_lines(data)
        self._transdata = utils.join_lines(transformed_data)
        self._k_signature = kernel_signature
        self._params = utils.join_lines(parameters)
        self._transparams = utils.join_lines(transformed_parameters)

        # misc
        self._model = utils.join_lines(model)
        self._args = utils.to_list(args)
        self._to_sample = utils.to_list(to_sample)
        self._data_priors = utils.to_list(data_priors)

        # append the kernel signature to transformed parameters
        sig = 'L = {0};'.format(self._k_signature)
        self._transparams = utils.join_strings([self._transparams, sig])


class StanModelMeanFunction(StanModelPart):
    """ Stan model mean functions. """
    @property
    def filename_component(self):
        return filenames.mean_part_filename(self._name)


class StanModelNoise(StanModelPart):
    """ Stan model noise models. """
    @property
    def filename_component(self):
        return filenames.noise_part_filename(self._name)


def none_to_list(arg: Optional[Any]) -> Any:
    """ If arg is None, turn it into an empty list. """
    if arg is None:
        return []
    return arg
