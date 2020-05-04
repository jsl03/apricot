# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------


def mean_part_filename(mean_type: str) -> str:
    """Assign mean function filename component from mean type"""
    if mean_type == 'linear':
        return 'linear_mean'
    if mean_type == 'zero':
        return 'zero_mean'
    msg = 'No filename part found for mean function "{0}"'.format(mean_type)
    raise RuntimeError(msg)


def noise_part_filename(noise_type: str) -> str:
    """Assign noise filename component from noise type"""
    if noise_type == 'infer':
        return 'infer_sigma'
    if noise_type == 'deterministic':
        return 'fixed_sigma'
    msg = 'No filename part found for noise type "{0}"'.format(noise_type)
    raise RuntimeError(msg)
