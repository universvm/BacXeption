import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from keras.models import model_from_json


def load_flat_model(path):

    json_file = open(path+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # Load model in Keras:
    loaded_model = model_from_json(loaded_model_json)

    # Load weights:
    loaded_model.load_weights(path+"model.h5")
    loaded_model.compile(loss='mean_squared_error', optimizer='adamax',
                         metrics=['accuracy'])

    print("Successfully loaded and compiled model from disk.")

    return loaded_model


def validate_flatness(X_test, y_test, loaded_model):
    """ Test labelled images with current model.
    Note: both arrays need to be numpy arrays and y_test has to be one hot encoded"""

    score = loaded_model.evaluate(X_test, y_test, verbose=0)

    # Print Evaluation:
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))


def predict_flatness(X_test):
    """ Uses the model to predict flatness. Must be given path argument. """

    # Temporarily in:
    # flat_path = 'result/resize-MSE-lowLoss/Seed_1231_a_94.44/'
    # flat_path = 'result/resize-ZeroPadding-CCE-96/'
    flat_path = 'result/crap_remover/95_not_all_data'
    # flat_path = 'result/non_resized_CCE_80/'

    # Checking whether the data path is a folder, if not, add "/"
    if flat_path[-1] != '/':
        flat_path = flat_path + '/'

    # Load model:
    loaded_model = load_flat_model(flat_path)

    # Validate:

    # Predict
    prediction = loaded_model.predict(X_test)

    return prediction


def outwrite_coords(coords_list, flat_results, data_path, filename):
    # Create output directory:
    output_directory = data_path + 'bac_detect_results/'
    os.makedirs(output_directory, exist_ok=True)

    # Open outfile:
    with open(output_directory + filename + '.txt', "w") as outfile:
        outfile.write('coords (x1,y1,width,height), flat_probability\n')

        # Saving coordinate and corresponding results on the same line:
        for (coords, result) in zip(coords_list, flat_results):
            outfile.write(str(coords) + ',' + str(round(result[0], 4)) + '\n')


def outwrite_graph(input_img, coords_list, flat_results, data_path, filename):
    """ Create graph with rectangles around identified bacteria."""

    # Plot image:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(input_img)

    # Add rectangles to image:
    for i in range(len(coords_list)):
        # Draw rectangle around segmented coins
        minr, minc, maxr, maxc = coords_list[i]  # y1, x1, width, height

        if flat_results[i][0] > 0.9:  # if flat
            color = 'green'
        else:  # else not flat:
            color = 'red'

        # Create Rectangles:
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor=color, linewidth=1,
                                  alpha=0.75)

        # Create label text:
        t = plt.text(minc+6, minr-8, 'flt: ' + "{0:.4f}".format(
            flat_results[i][0]), fontsize=6)
        t.set_bbox(dict(facecolor=color, alpha=0.75, edgecolor=color))
        ax.add_patch(rect)

    # Create output directory if not present:
    output_directory = data_path + 'bac_detect_results/'
    os.makedirs(output_directory, exist_ok=True)

    # Save graph:
    plt.savefig(output_directory + filename + '.png')

    # Display graph:
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def main_bac_detect(data_path, output_graph, output_coordinates):

    """ Main function which calls all the others.
    output_graph is a boolean, if True outputs the input image with rectangled sections.
    output_coordinates is a boolean, if True outputs the coordinates of the bacteria (divided into flat and not_flat."""

    # Checking whether the data path is a folder, if not, add "/"
    if data_path[-1] != '/':
        data_path = data_path + '/'

    # load all the images in folder:
    print('Loading data from ' + data_path)

    # Loop through all images:
    for filename in os.listdir(data_path):

        if filename.startswith('.'):  # Filter System Files:
            continue

        elif os.path.isfile(data_path+filename):  # Select files only
            # Clean image:
            clean_img, original_img = load_bac_pics(filename, data_path)

            # Extract rectangles in image:
            coords_list, cutouts_list = bac_rectify(clean_img, original_img)

            # Pre-Process:
            cutouts_list = cutout_preprocess(cutouts_list)

            # Main Neural Net:
            flat_results = predict_flatness(cutouts_list)
            # NOTE: flat_results stores one-hot encoded result in the same order as in coords list

            # Extract file name without extension:
            filename = filename.split('.')[0]

            # Output Coordinates if True:
            if output_coordinates:
                outwrite_coords(coords_list, flat_results, data_path, filename)

            if output_graph:
                outwrite_graph(original_img, coords_list, flat_results, data_path, filename)
