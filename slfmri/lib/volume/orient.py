from ..errors import *
from scipy.ndimage import affine_transform


def reorient_to_ras(data, affine, resol):
    """ Reorient and re-sample the input data into RAS space
    to ensure consistent orientation regardless of which axis
    was sliced during data acquisition.

    Returns:
        ras_data
        ras_resol
    """
    np.set_printoptions(precision=4, suppress=True)
    shape = data.shape
    fov_size = np.asarray(shape) * resol
    rotate_mat = (affine[:3, :3] / resol).astype(np.int8)
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

    ras_data = affine_transform(data, rotate, shift, output_shape=ras_shape)
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
