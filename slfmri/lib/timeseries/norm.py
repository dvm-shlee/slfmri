# from ..errors import *
# functions contains space normalization


# signal normalization tools
def demean(signal, axis=0):
    return signal - signal.mean(axis)


def standardization(signal, axis=0):
    demeaned = demean(signal, axis=axis)
    
    if demeaned.std(axis) == 0:
        return demeaned * 0
    else:
        norm_data = demeaned / demeaned.std(axis)
    return norm_data


def calc_modenorm(signal, mean, mode):
    return (signal - mean) * mode / mean + mode
