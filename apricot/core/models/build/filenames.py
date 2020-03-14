def mean_part_filename(mean_type):
    """Assign mean function filename component from mean type"""
    if mean_type == 'linear':
        return 'lm'
    elif mean_type == 'zero':
        return 'zm'
    else:
        raise RuntimeError('No filename part for mean function "{0}"'.
                           format(mean_type))


def noise_part_filename(noise_type):
    """Assign noise filename component from noise type"""
    if noise_type == 'infer':
        return 'ixi'
    elif noise_type == 'deterministic':
        return 'dxi'
    else:
        raise RuntimeError('No filename part for noise type "{0}"'.
                           format(noise_type))
