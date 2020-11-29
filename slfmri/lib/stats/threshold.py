from ..errors import *
from ..io.orient import decomp_dataobj, save_to_nib
from scipy.stats import norm


def thr_by_z(nib_obj, pval, twosided=False):
    data, affine, resol = decomp_dataobj(nib_obj)
    if len(nib_obj.shape) == 3:
        data = data[..., np.newaxis]
    thr_data = np.zeros(data.shape)
    for i in range(data.shape[-1]):
        d = data[..., i]
        do = np.zeros(d.shape)
        msk_idx = np.nonzero(d)
        if msk_idx[0].shape[0] == 0:
            pass
        else:
            dm = d[msk_idx]
            d_z = (dm - dm.mean()) / dm.std()
            if twosided:
                d_z[abs(d_z) < norm.ppf(1 - pval / 2)] = 0
            else:
                d_z[d_z < norm.ppf(1 - pval)] = 0
            do[msk_idx] = d_z
            thr_data[..., i] = do
    if len(nib_obj.shape) == 3:
        thr_data = thr_data[:, :, :, 0]
    return save_to_nib(thr_data, affine)


def thr_by_rank(nibobj, p):
    data, affine, resol = decomp_dataobj(nibobj)

    output = np.zeros(data.shape)
    midx = np.nonzero(data)
    mdata = data[midx]
    odata = mdata.copy()

    N = mdata.shape[0]
    P = int(N * p)
    odata[np.argsort(mdata)[:N - P]] = 0

    output[midx] = odata
    return save_to_nib(output, affine)


def thr_by_voxelsize(nib_obj, thr_size, nn_level=2):
    data, affine, resol = decomp_dataobj(nib_obj)

    pos_data = (data > 0).astype(np.int16)
    neg_data = (data < 0).astype(np.int16)

    pos_data_coords = np.transpose(np.nonzero(pos_data))
    neg_data_coords = np.transpose(np.nonzero(neg_data))

    n_pos = len(pos_data_coords)
    n_neg = len(neg_data_coords)

    from sklearn.cluster import DBSCAN
    est = DBSCAN(eps=np.sqrt(nn_level), min_samples=1)

    if n_pos:
        pos_lbls = est.fit(pos_data_coords).labels_
    else:
        pos_lbls = []
    if n_neg:
        neg_lbls = est.fit(neg_data_coords).labels_
    else:
        neg_lbls = []

    # filter clusters label
    pos_svv_lbls = []
    neg_svv_lbls = []

    if len(pos_lbls):
        for l in set(pos_lbls):
            sz = len(np.nonzero(pos_lbls == l)[0])
            if sz > thr_size:
                pos_svv_lbls.append(l)
    if len(neg_lbls):
        for l in set(neg_lbls):
            sz = len(np.nonzero(neg_lbls == l)[0])
            if sz > thr_size:
                neg_svv_lbls.append(l)

                # filter survives voxel by coordinates
    data_coords_flt = []
    for i, l in enumerate(pos_lbls):
        if l in pos_svv_lbls:
            data_coords_flt.append(pos_data_coords[i, :])
    for i, l in enumerate(neg_lbls):
        if l in neg_svv_lbls:
            data_coords_flt.append(neg_data_coords[i, :])

    # generate binarty mask
    data_flt = np.zeros(data.shape)
    flt_idx = tuple(np.array(data_coords_flt).astype(np.int16).T)
    data_flt[flt_idx] = data[flt_idx]
    return save_to_nib(data_flt, affine)