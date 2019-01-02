import os

from skimage.io import imread

from bac_detection.image_processing import thresholding_img
from bac_detection.image_processing import rectify_bac
from bac_detection.image_processing import preprocess_cutout
from bac_detection.output import outwrite_coords, outwrite_graph

from predict import predict_flatness

if __name__ == '__main__':
    # TODO: create config.py
    PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(PROJECT_ROOT_DIR,
                             'bacxeption/neural_network/test/')
    output_graph = True
    output_coordinates = True

    print(f'Loading data from {data_path}')

    id_dict = {}

    # Loop through all images:
    for filename in os.listdir(data_path):
        # TODO: variable in config
        if filename.endswith('.tiff') or filename.endswith('.tif'):
            print(f'Processing {filename}')
            # TODO: variable in config
            bac_img = imread(os.path.join(data_path, filename), as_gray=True)
            thresh_img = thresholding_img(bac_img)

            # Extract rectangles in image:
            coords_list, cutouts_list = rectify_bac(thresh_img, bac_img)

            # Normalize cutouts for network dimensions:
            norm_cutouts = preprocess_cutout(cutouts_list)
            flat_results = predict_flatness(norm_cutouts)

            filename = filename.split('.')[0]
            # Output Coordinates if True:
            if output_coordinates:
                outwrite_coords(coords_list, flat_results, data_path, filename)
            if output_graph:
                outwrite_graph(bac_img, coords_list, flat_results,
                               data_path, filename)
