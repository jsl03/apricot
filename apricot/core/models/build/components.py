from apricot.core.utils import (
    join_strings,
    join_lines,
    to_list,
)

from apricot.core.models.build.filenames import (
    noise_part_filename,
    mean_part_filename,
)

class StanModelPart:

    def __init__(self, name=None, functions=None, data=None, transformed_data=None,
                 parameters=None, transformed_parameters=None, model=None, args=None,
                 to_sample=None, data_priors=None):

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
        self._functions = join_lines(functions)
        self._data = join_lines(data)
        self._transdata = join_lines(transformed_data)
        self._params = join_lines(parameters)
        self._transparams = join_lines(transformed_parameters)
        self._model = join_lines(model)

        # misc
        self._args = to_list(args)
        self._to_sample = to_list(to_sample)
        self._data_priors = to_list(data_priors)

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
        self._functions = join_lines(functions)
        self._data = join_lines(data)
        self._transdata = join_lines(transformed_data)
        self._k_signature = kernel_signature
        self._params = join_lines(parameters)
        self._transparams = join_lines(transformed_parameters)

        # misc
        self._model = join_lines(model)
        self._args = to_list(args)
        self._to_sample = to_list(to_sample)
        self._data_priors = to_list(data_priors)

        # append the kernel signature to transformed parameters
        sig = 'L = {0};'.format(self._k_signature)
        self._transparams = join_strings([self._transparams, sig])

class StanModelMeanFunction(StanModelPart):
    @property
    def filename_component(self):
        return mean_part_filename(self._name)

class StanModelNoise(StanModelPart):
    @property
    def filename_component(self):
        return noise_part_filename(self._name)

def _none_to_list(x):
    """If x is None, turn it into an empty list"""
    if x is None:
        return []
    else:
        return x
