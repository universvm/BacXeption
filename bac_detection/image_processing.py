import numpy as np

from skimage import transform
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border


def normalize_img(input_img):
    """
    :param input_img: input numpy array of bac img
    :return normalized_image: numpy array with rescaled intensity
    """

    normalized_image = rescale_intensity(input_img, in_range=(
        input_img.min(), input_img.max()))

    return normalized_image


def thresholding_img(bac_image):
    """
    :param bac_image: numpy array of bac image
    :return thresh_image: numpy array with Otsu thresholding
    """

    # Apply threshold:
    thresh = threshold_otsu(bac_image)
    bw = closing(bac_image > thresh, square(3))

    # Remove artifacts connected to image border:
    thresh_image = clear_border(bw)

    return thresh_image


def rectify_bac(image_array, original_img, input_dim, min_px_area,
                extra_border_px):
    """
    :param image_array: Array of image with thresholding
    :param original_img: Original array
    :param input_dim: Tuple image dimension for network
    :param min_px_area: Int of px for minimum area of a bacterium
    :param extra_border_px: Int of extra px added to the cutout
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
        if rect.area >= min_px_area:
            x1, y1, width, height = rect.bbox
            min_axval = min(rect.bbox)
            # Select values from image:
            if min_axval - extra_border_px > 0:  # Take extra px if available
                output_bac = original_img[x1 - extra_border_px:width +
                                                               extra_border_px,
                                          y1 - extra_border_px:height +
                                                               extra_border_px]
            else:
                output_bac = original_img[x1:width, y1:height]

            output_bac = transform.resize(output_bac, input_dim)

            coords_list.append(rect.bbox)
            cutouts_list.append(output_bac)

    return coords_list, cutouts_list


def preprocess_cutout(cutouts_list, input_dim):
    """ Necessary cutouts pre-processing for Neural Network
    :param cutouts_list: Python list with bacteria cutouts
    :param input_dim: tuple of imgs dimension
    :return norm_cutouts: 4-d numpy array (len(cutouts_list), input_dim)
    """

    # Convert to numpy array:
    norm_cutouts = np.array(cutouts_list)
    # Reshape numpy array:
    norm_cutouts = norm_cutouts.reshape(len(cutouts_list), *input_dim)

    return norm_cutouts
