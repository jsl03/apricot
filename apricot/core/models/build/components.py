import typing
from apricot.core import utils
from apricot.core.models.build import filenames

Model_Code_Type = typing.Optional[typing.Union[typing.List[str], str]]

#TODO tidy this up; some options are deprecated (eg. is args still used?)
class StanModelPart:

    def __init__(self,
                 name : typing.Optional[str] = None,
                 functions : Model_Code_Type = None,
                 data : Model_Code_Type = None,
                 transformed_data : Model_Code_Type = None,
                 parameters : Model_Code_Type = None,
                 transformed_parameters : Model_Code_Type = None,
                 model : Model_Code_Type = None,
                 args : Model_Code_Type = None,
                 to_sample : Model_Code_Type = None,
                 data_priors : Model_Code_Type = None,
    ):

        self._name = name

        # avoiding using mutable defaults
        functions = _none_to_list(functions)
        data = _none_to_list(data)
        transformed_data = _none_to_list(transformed_data)
        paramaters = _none_to_list(parameters)
        transformed_parameters = _none_to_list(transformed_parameters)
        model = _none_to_list(model)
        args = _none_to_list(args)
        to_sample = _none_to_list(to_sample)
        data_prios = _none_to_list(data_priors)

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
        return self._name

    @property
    def functions(self):
        return self._functions

    @property
    def data(self):
        return self._data

    @property
    def transdata(self):
        return self._transdata

    @property
    def params(self):
        return self._params

    @property
    def transparams(self):
        return self._transparams

    @property
    def model(self):
        return self._model

    @property
    def args(self):
        return self._args

    @property
    def to_sample(self):
        return self._to_sample

    @property
    def data_priors(self):
        return self._data_priors

    def __iter__(self):
        # calling iter on the class accesses a generator yielding the model
        # code blocks, for example:
        # >>> for block in model_component:
        # >>>     <build the model>
        blockGen = (c for c in (
            self.functions,
            self.data,
            self.transdata,
            self.params,
            self.transparams,
            self.model))
        return blockGen

class StanModelKernel(StanModelPart):

    def __init__(self, name=None, functions=None, data=None,
                 transformed_data=None, parameters=None,
                 transformed_parameters=None, model=None, kernel_signature=None,
                 args=None, to_sample=None, data_priors=None):

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
    @property
    def filename_component(self):
        return filenames.mean_part_filename(self._name)

class StanModelNoise(StanModelPart):
    @property
    def filename_component(self):
        return filenames.noise_part_filename(self._name)

def _none_to_list(x):
    """ If x is None, turn it into an empty list. """
    if x is None:
        return []
    else:
        return x
