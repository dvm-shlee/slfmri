import re
import numpy as np
from .methods import *
from .messages import *
from nibabel import Nifti1Image, affines


class ImageObj(Nifti1Image):
    """ ImageObject for PyNIT
    """
    # def __init__(self):
    #     super(ImageObj, self).__init__()

    def save_as(self, filename, quiet=False):
        """ Save as a new file with current affine information
        """
        self.header['sform_code'] = 0
        self.header['qform_code'] = 1
        self.to_filename('{}'.format(filename))
        if not quiet:
            print("NifTi1 format image is saved to '{}'".format(filename))


def load(filename):
    """ load available file
    available exts: .nii(.gz), .mga, .xls(x), .csv, .tsv, .json
    :param filename: file want to load
    :type filename: str
    :return: object
    """
    if '.nii' in filename:
        img = ImageObj.load(filename)
    elif '.mha' in filename:
        try:
            import SimpleITK as sitk
            mha = sitk.ReadImage(filename)
        except:
            raise ImportItkFailure
        data = sitk.GetArrayFromImage(mha)
        resol = mha.GetSpacing()
        origin = mha.GetOrigin()
        affine = affines.from_matvec(np.diag(resol), origin)
        img = ImageObj(data, affine)
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
            raise InputPathError
    return img



def parsing_atlas(path):
    """Parsing atlas imageobj and label
    :param path:
    :return:
    """
    label = dict()
    affine = list()
    if os.path.isdir(path):
        atlasdata = None
        list_of_rois = sorted([img for img in os.listdir(path) if '.nii' in img])
        rgbs = np.random.rand(len(list_of_rois), 3)
        label[0] = 'Clear Label', [.0, .0, .0]

        for idx, img in enumerate(list_of_rois):
            imageobj = ImageObj.load(os.path.join(path, img))
            affine.append(imageobj.affine)
            if not idx:
                atlasdata = np.asarray(imageobj.dataobj)
            else:
                atlasdata += np.asarray(imageobj.dataobj) * (idx + 1)
            label[idx+1] = splitnifti(img), rgbs[idx]
        atlas = ImageObj(atlasdata, affine[0])

    elif os.path.isfile(path):
        atlas = ImageObj.load(path)
        if '.nii' in path:
            filepath = os.path.basename(splitnifti(path))
            dirname = os.path.dirname(path)
            if dirname == '':
                dirname = '.'
            for f in os.listdir(dirname):
                if filepath in f:
                    if '.lbl' in f:
                        filepath = os.path.join(dirname, "{}.lbl".format(filepath))
                    elif '.label' in f:
                        filepath = os.path.join(dirname, "{}.label".format(filepath))
                    else:
                        filepath = filepath
            if filepath == os.path.basename(splitnifti(path)):
                raise NoLabelFile
        else:
            raise NoLabelFile
        pattern = r'^\s+(?P<idx>\d+)\s+(?P<R>\d+)\s+(?P<G>\d+)\s+(?P<B>\d+)\s+' \
                  r'(\d+|\d+\.\d+)\s+\d+\s+\d+\s+"(?P<roi>.*)$'
        with open(filepath, 'r') as labelfile:
            for line in labelfile:
                if re.match(pattern, line):
                    idx = int(re.sub(pattern, r'\g<idx>', line))
                    roi = re.sub(pattern, r'\g<roi>', line)
                    roi = roi.split('"')[0]
                    rgb = re.sub(pattern, r"\g<R> \g<G> \g<B>", line)
                    rgb = rgb.split()
                    rgb = np.array([c for c in map(float, rgb)])/255
                    label[idx] = roi, rgb
    else:
        raise InputPathError
    data = np.asarray(atlas.dataobj)
    return atlas, label


def save_label(label, filename):
    """ Save label instance to file
    :param label:
    :param filename:
    :return:
    """
    with open(filename, 'w') as f:
        line = list()
        for idx in label.keys():
            roi, rgb = label[idx]
            rgb = np.array(rgb) * 255
            rgb = rgb.astype(int)
            if idx == 0:
                line = '{:>5}   {:>3}  {:>3}  {:>3}        0  0  0    "{}"\n'.format(idx, rgb[0], rgb[1], rgb[2], roi)
            else:
                line = '{}{:>5}   {:>3}  {:>3}  {:>3}        1  1  0    "{}"\n'.format(line, idx, rgb[0], rgb[1], rgb[2], roi)
        f.write(line)


class Atlas(object):
    """ This class templating the segmentation image object to handle atlas related attributes
    """

    _ext = '.nii.gz'

    def __init__(self, path=None):
        self.path = path
        self._label = None
        self._coordinates = None
        if type(path) is ImageObj:
            self._image = path
        elif type(path) is str:
            self.load(path)
        else:
            raise InputFileError

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def image(self):
        return self._image

    @property
    def label(self):
        return self._label

    @image.setter
    def image(self, imageobj):
        if type(imageobj) is ImageObj:
            self._image = imageobj
        else:
            raise InputObjectError

    def load(self, path):
        self._image, self._label = parsing_atlas(path)

    def save_as(self, filename, label_only=False, quiet=False):
        if not label_only:
            self._image.save_as("{}{}".format(filename, self._ext), quiet=quiet)
        save_label(self._label, "{}.label".format(filename))

    def extract(self, path, contra=False, merge=False, surfix=None):
        if not os.path.exists(path):
            # try:
            mkdir(path)
            # except:
            #     raise messages.InputPathError
        num_of_rois = int(np.max(self._image._dataobj))
        for i in range(num_of_rois + 1):
            if not i:
                pass
            else:
                try:
                    label, roi = self[i]
                    if contra:
                        label = 'contra_' + label
                        roi._dataobj = roi._dataobj[::-1, ...]
                    else:
                        if merge:
                            label = 'bilateral_' + label
                            roi._dataobj += roi._dataobj[::-1, ...]
                            roi._dataobj[roi._dataobj > 0] = 1
                    if surfix:
                        label = "{}_{}".format(surfix, label)
                    roi.to_filename(os.path.join(path, "{}{}".format(label, self._ext)))
                except:
                    pass

    def __getitem__(self, idx):
        if not self._image:
            return None
        else:
            if idx != 0:
                mask = np.zeros(self.image.shape)
                mask[self.image.get_data() == idx] = 1
                maskobj = ImageObj(mask, self.image.affine)
                return self.label[idx][0], maskobj
            else:
                return None

    def __repr__(self):
        labels = None
        for idx in self.label.keys():
            if not idx:
                labels = '[{:>3}] {:>40}\n'.format(idx, self.label[idx][0])
            else:
                labels = '{}[{:>3}] {:>40}\n'.format(labels, idx, self.label[idx][0])
        return labels