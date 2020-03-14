import re
import typing
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np


sns_context = 'paper'
sns_style = 'white'
sns_palette = 'magma'


sns.set_context(sns_context)
sns.set_style(sns_style)
sns.set_palette(sns_palette)


Figure_Size = typing.Optional[typing.Tuple[float, float]]


def _assign_param_name(name: str, ndim: int, dim: int, log: bool):
    """ Assign parameter name for plotting. """
    if log:
        _name = 'log {0}'.format(name)
    else:
        _name = name
    if ndim > 1:
        param_name = _name + ' [{0}]'.format(dim)
    else:
        param_name = _name
    return param_name


def _parse_param_str(s: str):
    """ Strip brackets from string s and return the bracketed value """
    x_ = re.sub('\[.*\]', '', s)
    match = re.search(r'\[(.*?)\]', s)
    if match:
        return x_, int(match.group(1))
    else:
        return x_, None


def _get_param(hyperparameters: dict, col: str):
    name, dim = _parse_param_str(col)
    if dim:  # parameter is vector valued
        return hyperparameters[name][:, dim-1]
    else:  # parameter is scalar valued
        return hyperparameters[name][:, 0]


def plot_parameter(
    hyperparameters: dict,
    name: str,
    info: dict,
    log_param: bool = False,
    figsize: Figure_Size = None,
):
    if figsize is None:
        figsize = (7,3)
    ndim = hyperparameters[name].shape[1]
    plt.figure(figsize=(figsize[0], figsize[1] * ndim))
    gs = gridspec.GridSpec(ndim, 2)
    for dim in range(ndim):
        param_name = _assign_param_name(name, ndim, dim, log_param)
        ax0 = plt.subplot(gs[dim, 0])
        ax1 = plt.subplot(gs[dim, 1])
        nchains = int(max(info['sample_chain_id']))
        for chain in range(1,nchains+1):
            data = hyperparameters[name][:, dim][info['sample_chain_id'] == chain]
            if log_param:
                data = np.log(data)
            sns.distplot(data, hist=False, kde=True, label='chain {0}'.format(chain), ax=ax0)
            plt.plot(data, label='chain {0}'.format(chain))
        ax0.legend(frameon=False)
        ax0.set_xlabel(param_name)
        ax0.set_ylabel('density')
        ax1.set_xlabel('iteration #')
        ax1.set_ylabel(param_name)
        ax1.set_xlim(0, data.shape[0]) 
    plt.tight_layout()


def plot_divergences(
    hyperparameters: dict,
    info: dict,
    figsize: Figure_Size = None,
):
    if figsize is None:
        figsize = (7, 3)
    colnames = info['colnames']
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    idx_true = info['divergent'] == 1
    idx_false = info['divergent'] == 0
    data_array = np.array([_get_param(hyperparameters, col) for col in colnames]).T
    col_mins = np.min(data_array, axis=0)
    col_maxs = np.max(data_array, axis=0)
    data_array_normed = (data_array - np.min(data_array, axis=0)) / (col_maxs - col_mins)
    nondivergent_transitions = data_array_normed[idx_false, :]
    divergent_transitions = data_array_normed[idx_true, :]

    ax.plot(nondivergent_transitions.T, color='.5', alpha=0.1)
    if divergent_transitions.shape[0] != 0:
        ax.plot(divergent_transitions.T, color='r', alpha=1.0)
    ax.set_xlim(0, len(colnames)-1)
    ax.set_xticks(range(0, len(colnames)))
    ax.set_xticklabels(colnames)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['min', 'max'])
    ax.set_xlabel('parameter')
    ax.set_ylabel('normalised value')
    plt.show()
