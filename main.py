from config import *

from skimage.io import imread

from bac_detection.image_processing import thresholding_img
from bac_detection.image_processing import rectify_bac
from bac_detection.image_processing import preprocess_cutout
from bac_detection.output import outwrite_coords, outwrite_graph
from bac_detection.predict import predict_flatness


if __name__ == '__main__':
    print(f'Loading data from {TEST_DIR}')

    # Loop through all test images:
    for filename in os.listdir(TEST_DIR):
        if filename.endswith(FORMAT_IMG):
            print(f'Processing {filename}')
            bac_img = imread(os.path.join(TEST_DIR, filename), as_gray=True)
            thresh_img = thresholding_img(bac_img)

            # Extract rectangles of bacteria in image:
            coords_list, cutouts_list = rectify_bac(thresh_img, bac_img)

            # Normalize cutouts for network dimensions:
            norm_cutouts = preprocess_cutout(cutouts_list)
            flat_results = predict_flatness(norm_cutouts, LOG_DIR,
                                            LOSS_FUNC,  OPTIMIZER, METRICS)

            filename = filename.split('.')[0]

            if OUTPUT_COORDS:
                outwrite_coords(coords_list, flat_results, TEST_DIR, filename)
            if OUTPUT_GRAPH:
                outwrite_graph(bac_img, coords_list, flat_results,
                               TEST_DIR, filename)
