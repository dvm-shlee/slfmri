def load(filename):
    """
    load available file
    available exts: .nii(.gz), .mha, .xls(x), .csv, .tsv, .json

    :param filename: file want to load
    :type filename: str
    :return: object
    """
    if '.nii' in filename:
        from nibabel import Nifti1Image
        img = Nifti1Image.load(filename)
    else:
        import pandas as pd
        if '.xls' in filename:
            img = pd.read_excel(filename)
        elif '.csv' in filename:
            img = pd.read_csv(filename)
        elif '.tsv' in filename:
            img = pd.read_table(filename)
        elif '.1D' in filename:
            img = pd.read_csv(filename, header=None, sep='\s+')
        elif '.json' in filename:
            import json
            img = json.load(open(filename))
        else:
            raise Exception('Input filetype is not compatible.')
    return img


def save_to_nii(func_img, niiobj, fpath):
    import nibabel as nib
    nii = nib.Nifti2Image(func_img, niiobj.affine)
    nii._header = niiobj._header
    nii.to_filename(fpath)