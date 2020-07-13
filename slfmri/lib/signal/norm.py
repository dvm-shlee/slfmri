def demean(signal, axis=0):
    """
    Demean signal

    Args:
        signal: input signal
        axis: axis to calculate mean value

    Returns:
        demeaned signal
    """
    return signal - signal.mean(axis)


def standardize(signal, axis=0):
    """
    Standardize signal

    Args:
        signal: input signal
        axis: axis to apply standardization

    Returns:
        standardized signal
    """
    demeaned = demean(signal, axis=axis)
    
    if demeaned.std(axis) == 0:
        return demeaned * 0
    else:
        norm_data = demeaned / demeaned.std(axis)
    return norm_data


def modenorm(signal, mean, mode):
    """
    Mode normalization
    Args:
        signal: input signal
        mean: mean signal
        mode: target mode

    Returns:
        modenormed signal
    """
    return (signal - mean) * mode / mean + mode
