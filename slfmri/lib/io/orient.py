from ..errors import *
import nibabel as nib
from scipy.ndimage import affine_transform


def decomp_dataobj(nib_img):
    data = np.asarray(nib_img.dataobj).copy()
    affine = nib_img.affine.copy()
    resol = nib_img.header['pixdim'][1:4]
    return data, affine, resol


def save_to_nib(data, affine):
    nii = nib.Nifti1Image(data, affine)
    nii.header['sform_code'] = 0
    nii.header['qform_code'] = 1
    return nii


def reorient_to_ras(data, affine, resol):
    """ Reorient and re-sample the input data into RAS space
    to ensure consistent orientation regardless of which axis
    was sliced during data acquisition.

    Returns:
        ras_data
        ras_resol
    """
    np.set_printoptions(precision=4, suppress=True)
    shape = data.shape[:3]
    fov_size = np.asarray(shape) * resol
    rotate_mat = np.round(affine[:3, :3] / resol, decimals=0).astype(np.int8)
    origin = rotate_mat.dot(fov_size/2)

    org_affine = affine.copy()
    org_affine[:3, 3] = -origin

    ras_resol = abs(rotate_mat.dot(resol))
    ras_shape = abs(rotate_mat.dot(shape))
    ras_affine = np.eye(4)
    ras_affine[:3, :3] = np.diag(resol)
    ras_affine[:3, 3] = -fov_size/2

    org_mm2vox = np.linalg.inv(org_affine)
    ras_mm2vox = org_mm2vox.dot(ras_affine)

    rotate = ras_mm2vox[:3, :3]
    shift = ras_mm2vox[:3, 3]

    if len(data.shape) == 4:
        ras_data = np.zeros(data.shape)
        for f in range(data.shape[-1]):
            ras_data[:,:,:, f] = affine_transform(data[:,:,:, f],
                                                  rotate, shift,
                                                  output_shape=ras_shape)
    else:
        ras_data = affine_transform(data, rotate, shift,
                                    output_shape=ras_shape)
    return ras_data, ras_resol


def determine_slice_plane(slice_axis, affine, resol):
    """ return the original scheme of slice plane """
    rotate_mat = (affine[:3, :3] / resol).astype(np.int8)
    ras_axis = abs(rotate_mat.dot(range(3))).tolist()
    return ['sagittal', 'coronal', 'axial'][ras_axis.index(slice_axis)]


def from_matvec(mat, vec):
    affine = np.eye(4)
    affine[:3,:3] = mat
    affine[:3, 3] = vec
    return affine


def to_matvec(matrix):
    return matrix[:3, :3], matrix[:3, 3]


def norm_orient(nib_img):
    """ Reorient and re-sample the input data into RAS space
    to ensure consistent orientation regardless of which axis
    was sliced during data acquisition.

    Returns:
        nib.NifTi1Image obj
    """
    from scipy.ndimage import affine_transform
    data, affine, resol = decomp_dataobj(nib_img)

    np.set_printoptions(precision=4, suppress=True)
    shape = data.shape
    fov_size = np.asarray(shape) * np.array(resol)
    rotate_mat = np.round(affine[:3, :3] / np.array(resol)).astype(int)

    origin = rotate_mat.dot(fov_size / 2)
    org_affine = affine.copy()
    org_affine[:3, 3] = -origin

    ras_resol = abs(rotate_mat.dot(resol))
    ras_shape = abs(rotate_mat.dot(shape))
    ras_affine = np.eye(4)
    ras_affine[:3, :3] = np.diag(ras_resol * np.array([1, -1, 1]))
    ras_affine[:3, 3] = -fov_size / 2 * np.array([1, -1, 1])

    org_mm2vox = np.linalg.inv(org_affine)
    ras_mm2vox = org_mm2vox.dot(ras_affine)

    rotate = ras_mm2vox[:3, :3]
    shift = ras_mm2vox[:3, 3]

    ras_data = affine_transform(data, rotate, shift, output_shape=ras_shape)
    ras_affine[:3, :3] = ras_affine[:3, :3]
    ras_affine[:3, 3] = rotate_mat.dot(affine[:3, 3]) * np.array([1, -1, 1])

    return save_to_nib(ras_data, ras_affine)


def pad_by_voxel(nib_img, a=None, p=None, l=None, r=None, i=None, s=None):
    """
    a: anterior
    p: posterior
    l: left
    r: right
    s: superior
    i: inferior
    """
    options = dict(a=a, p=p, l=l, r=r, i=i, s=s)
    data, affine, resol = decomp_dataobj(nib_img)
    x, y, z = data.shape
    for k, o in options.items():
        if o is None:
            pass
        else:
            if k == 'a':
                affine[1, 3] += resol[1] * o
                pad = np.zeros([x, o, z])
                data = np.concatenate([pad, data], axis=1)
                x, y, z = data.shape
            if k == 'p':
                pad = np.zeros([x, o, z])
                data = np.concatenate([data, pad], axis=1)
                x, y, z = data.shape
            if k == 'l':
                affine[0, 3] += resol[0] * o
                pad = np.zeros([o, y, z])
                data = np.concatenate([pad, data], axis=0)
                x, y, z = data.shape
            if k == 'r':
                pad = np.zeros([o, y, z])
                data = np.concatenate([data, pad], axis=0)
                x, y, z = data.shape
            if k == 'i':
                affine[2, 3] -= resol[2] * o
                pad = np.zeros([x, y, o])
                data = np.concatenate([pad, data], axis=2)
                x, y, z = data.shape
            if k == 's':
                pad = np.zeros([x, y, o])
                data = np.concatenate([data, pad], axis=2)
                x, y, z = data.shape
    return save_to_nib(data, affine)


def crop_by_voxel(nib_img, a=None, p=None, l=None, r=None, i=None, s=None):
    """
    a: anterior
    p: posterior
    l: left
    r: right
    s: superior
    i: inferior
    """
    options = dict(a=a, p=p, l=l, r=r, i=i, s=s)
    data, affine, resol = decomp_dataobj(nib_img)
    x, y, z = data.shape
    for k, o in options.items():
        if o is None:
            pass
        else:
            if k == 'a':
                affine[1, 3] += resol[1] * o
                data = data[:, o:, :]
                x, y, z = data.shape
            if k == 'p':
                data = data[:, :(y - o), :]
                x, y, z = data.shape
            if k == 'l':
                affine[0, 3] += resol[0] * o
                data = data[o:, :, :]
                x, y, z = data.shape
            if k == 'r':
                data = data[:(x - o), :, :]
                x, y, z = data.shape
            if k == 'i':
                affine[2, 3] += resol[2] * o
                data = data[:, :, o:]
                x, y, z = data.shape
            if k == 's':
                data = data[:, :, :(z - o)]
                x, y, z = data.shape
    return save_to_nib(data, affine)


def concat_3d_to_4d(*nib_objs):
    concat_data = []
    affine = None
    for nii in nib_objs:
        data, affine, _ = decomp_dataobj(nii)
        concat_data.append(data[..., np.newaxis])
    concat_data = np.concatenate(concat_data, axis=-1)
    return save_to_nib(concat_data, affine)
