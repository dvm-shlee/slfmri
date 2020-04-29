def tsnr(signal):
    return signal.mean() / signal.std()


def dvars(func_img, mask_img=None):
    import numpy as np
    if mask_img is not None:
        indice = np.nonzero(mask_img)
    else:
        indice = np.nonzero(func_img.mean(-1))
    diff_img = np.diff(func_img[indice], axis=-1)
    dvars = np.sqrt(np.square(diff_img).mean(0))
    return np.insert(dvars, 0, np.nan)


def bold_meanstd(func_img, mask_img=None):
    import numpy as np
    from .tools import get_funcobj, apply_funcobj
    from .norm import demean

    if mask_img is not None:
        indice = np.nonzero(mask_img)
    else:
        indice = np.nonzero(func_img.mean(-1))
    demean_obj = get_funcobj(demean, axis=-1)
    demeaned_img = apply_funcobj(demean_obj, func_img, mask_img)

    return demeaned_img[indice].mean(0), demeaned_img[indice].std(0)


def mparam_fd(volreg):
    """ Framewise displacement """
    import numpy as np
    return np.abs(np.insert(np.diff(volreg, axis=0), 0, 0, axis=0)).sum(axis=1)


def mparam_ard(volreg):
    """ Absolute rotational displacement """
    import numpy as np
    return np.abs(np.insert(np.diff(volreg[volreg.columns[:3]], axis=0),
                            0, 0, axis=0)).sum(axis=1)


def mparam_atd(volreg):
    import numpy as np
    return np.abs(np.insert(np.diff(volreg[volreg.columns[3:]], axis=0),
                            0, 0, axis=0)).sum(axis=1)




