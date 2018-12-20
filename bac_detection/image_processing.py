import numpy as np

from skimage import transform
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border


def thresholding_img(bac_image):
    """
    :param bac_image: numpy array of bac image
    :return thresh_image: numpy array with Otsu thresholding
    """
    # TODO: Hacky
    # input_bac = exposure.rescale_intensity(input_bac, in_range=(
    #     input_bac.min(), input_bac.max()))
    bac_image = bac_image - bac_image.min()/1.05

    # Apply threshold:
    thresh = threshold_otsu(bac_image)
    bw = closing(bac_image > thresh, square(3))

    # Remove artifacts connected to image border:
    thresh_image = clear_border(bw)

    return thresh_image


def rectify_bac(image_array, original_img):
    """
    :param image_array: Array of image with thresholding
    :param original_img: Original array
    :return coords_list: Array of integers (format x1, y1, width, height)
    :return cutouts_list: Array of bacteria cutouts from the original image
    """

    coords_list = []
    cutouts_list = []

    # Create rectangles:
    label_image = label(image_array)
    rects_list = regionprops(label_image)

    # Iterate through selected bac
    for rect in rects_list:
        if rect.area >= 50:
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


def preprocess_cutout(cutouts_list):
    """ Necessary cutouts pre-processing for Neural Network
    :param cutouts_list: Python list with bacteria cutouts
    :return norm_cutouts: 4-d numpy array (n, 180, 180, 1)
    """

    # Convert to numpy array:
    norm_cutouts = np.array(cutouts_list)
    # Reshape numpy array:
    norm_cutouts = norm_cutouts.reshape(len(cutouts_list), 180, 180, 1)
    # Normalize data:
    norm_cutouts = np.array(norm_cutouts).astype('float64')

    return norm_cutouts
