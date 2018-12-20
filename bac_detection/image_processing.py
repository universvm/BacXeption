import numpy as np

from skimage import io
from skimage import transform
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border


def load_images(filename, data_folder):
    """ Load + Pre-process bacterial images from folder.
    :param filename: str of image filename
    :param data_folder: folder_path to filename
    :return thresh_image: image array with
    """
    print(f'Thresholding {filename}...')

    input_bac = io.imread(data_folder + filename, flatten=True)
    # input_bac = exposure.rescale_intensity(input_bac, in_range=(
    #     input_bac.min(), input_bac.max()))

    input_bac = input_bac - input_bac.min()/1.05

    # Apply threshold:
    thresh = threshold_otsu(input_bac)
    bw = closing(input_bac > thresh, square(3))

    # Remove artifacts connected to image border:
    thresh_image = clear_border(bw)

    return thresh_image, input_bac


def preprocess_cutout(cutouts_list):
    """ Necessary cutouts pre-processing for Neural Network
    :param cutouts_list: Python list with bacteria cutouts
    :return normalized_cutouts: 4-d numpy array (n, 180, 180, 1)
    """

    # Convert to numpy array:
    normalized_cutouts = np.array(cutouts_list)

    # Reshape numpy array:
    normalized_cutouts = normalized_cutouts.reshape(len(cutouts_list), 180, 180, 1)

    # Normalize data:
    normalized_cutouts = np.array(normalized_cutouts).astype('float64')

    return normalized_cutouts


def rectify_bac_img(image_array, original_img):
    """

    :param image_array: Array of image with thresholding
    :param original_img: Original array
    :return: coords_list: Array of integers (format x1, y1, width, height)
    :return: cutouts_list: Array of bacteria cutouts from the original image
    """

    # Create rectangles:
    label_image = label(image_array)
    rects_list = regionprops(label_image)

    # array with coords of bacteria:
    coords_list = []  # format x1, y1, width, height

    # array with all the cutouts of bacteria in image_array:
    cutouts_list = []

    # Iterate through selected bac
    for rect in rects_list:

        if rect.area >= 50:

            # Extracting coordinates:
            x1, y1, width, height = rect.bbox
            min_axval = min(rect.bbox)

            # Select values from image:
            if min_axval - 5 > 0:  # Take extra px if available
                output_bac = original_img[x1 - 5:width + 5, y1 - 5:height + 5]
            else:
                output_bac = original_img[x1:width, y1:height]

            # Normalize:
            output_bac = output_bac / np.max(output_bac)

            # Resize multiplier:
            output_bac = transform.resize(output_bac, (180, 180))

            coords_list.append(rect.bbox)
            cutouts_list.append(output_bac)

    return coords_list, cutouts_list
