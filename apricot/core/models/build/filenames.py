# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
# ------------------------------------------------------------------------------


def mean_part_filename(mean_type: str) -> str:
    """Assign mean function filename component from mean type"""
    if mean_type == 'linear':
        return 'lm'
    elif mean_type == 'zero':
        return 'zm'
    else:
        raise RuntimeError('No filename part found for mean function "{0}"'.
                           format(mean_type))


def noise_part_filename(noise_type: str) -> str:
    """Assign noise filename component from noise type"""
    if noise_type == 'infer':
        return 'ixi'
    elif noise_type == 'deterministic':
        return 'dxi'
    else:
        raise RuntimeError('No filename part found for noise type "{0}"'.
                           format(noise_type))
