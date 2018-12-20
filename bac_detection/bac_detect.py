# Modules:
from skimage.filters import threshold_otsu
from skimage import exposure
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage import io
from skimage import transform
import numpy as np
import os


def rectify_bac_img(image_array, original_img):
    """

    :param image_array: Array of image with thresholding
    :param original_img: Original array
    :return: coords_list: Array of integers (format x1, y1, width, height)
            cutouts_list: Array of bacteria cutouts from the original image
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
