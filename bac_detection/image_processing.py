import numpy as np

from skimage import io
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import clear_border


def load_images(filename, data_folder):
    """ Load + Pre-process bacterial images from folder. """

    # TODO: Filtering system file:
    # if not filename.startswith('.'):

    input_bac = io.imread(data_folder + filename, flatten=True)
    # input_bac = exposure.rescale_intensity(input_bac, in_range=(
    #     input_bac.min(), input_bac.max()))

    input_bac = input_bac - input_bac.min()/1.05

    # Apply threshold:
    thresh = threshold_otsu(input_bac)
    bw = closing(input_bac > thresh, square(3))

    # Remove artifacts connected to image border:
    cleared = clear_border(bw)

    print('Image ' + filename + ' imported and cleaned successfully!')

    return cleared, input_bac


def preprocess_cutout(cutouts_list):
    """ Necessary cutouts pre-processing for Neural Network """

    # Convert to numpy array:
    cutouts_list = np.array(cutouts_list)

    # Reshape numpy array:
    cutouts_list = cutouts_list.reshape(len(cutouts_list), 180, 180, 1)

    # Normalize data:
    cutouts_list = np.array(cutouts_list).astype('float64')

    return cutouts_list
