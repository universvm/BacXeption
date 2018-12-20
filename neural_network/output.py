import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def outwrite_coords(coords_list, flat_results, data_path, filename):
    """
    :param coords_list: int array (y1, x1, width, height)
    :param flat_results: array of float (probability)
    :param data_path: str path to file
    :param filename: str file name
    """
    # TODO: Refactor magic variable
    output_directory = data_path + 'results/'
    os.makedirs(output_directory, exist_ok=True)

    # Open outfile:
    # TODO: Refactor magic variable
    with open(output_directory + filename + '.txt', "w") as outfile:
        outfile.write('coords (x1,y1,width,height), flat_probability\n')

        # Saving coordinate and corresponding results on the same line:
        for (coords, result) in zip(coords_list, flat_results):
            outfile.write(str(coords) + ',' + str(round(result[0], 4)) + '\n')


def outwrite_graph(input_img, coords_list, flat_results, data_path, filename):
    """
    Create graph with rectangles around identified bacteria.
    :param input_img: original image array
    :param coords_list: int array (y1, x1, width, height)
    :param flat_results: array of float (probability)
    :param data_path: str path to file
    :param filename: str file name
    """

    # Plot image:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(input_img)

    # Add rectangles to image:
    for i in range(len(coords_list)):
        minr, minc, maxr, maxc = coords_list[i]  # y1, x1, width, height

        # TODO: Refactor magic variable
        if flat_results[i][0] > 0.9:  # if flat
            color = 'green'
        else:  # else not flat:
            color = 'red'

        # Create Rectangles:
        # TODO: Refactor magic variable
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor=color, linewidth=1,
                                  alpha=0.75)

        # Create label text:
        # TODO: Refactor magic variable
        t = plt.text(minc+6, minr-8, 'flt: ' + "{0:.4f}".format(
            flat_results[i][0]), fontsize=6)
        t.set_bbox(dict(facecolor=color, alpha=0.75, edgecolor=color))
        ax.add_patch(rect)

    # Create output directory if not present:
    # TODO: Refactor magic variable
    output_directory = data_path + 'results/'
    os.makedirs(output_directory, exist_ok=True)

    # Save graph:
    # TODO: Refactor magic variable
    plt.savefig(output_directory + filename + '.png')

    # Display graph:
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
