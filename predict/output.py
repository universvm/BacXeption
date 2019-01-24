import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def outwrite_coords(coords_list, flat_results, out_path, filename):
    """
    :param coords_list: int array (y1, x1, width, height)
    :param flat_results: array of float (probability)
    :param out_path: str path to output
    :param filename: str file name
    """

    # Open outfile:
    with open(os.path.join(out_path, filename + '.txt'), 'w') as outfile:
        outfile.write('coords (x1, y1, width, height), probability\n')

        # Saving coordinate and corresponding results on the same line:
        for (coords, result) in zip(coords_list, flat_results):
            outfile.write(str(coords) + ',' + str(round(result[0], 4)) + '\n')


def outwrite_graph(input_img, coords_list, flat_results, out_path, filename,
                   acceptable_threshold, display_graph):
    """
    Create graph with rectangles around identified bacteria.
    :param input_img: original image array
    :param coords_list: int array (y1, x1, width, height)
    :param flat_results: array of float (probability)
    :param out_path: str path to output
    :param filename: str file name
    :param acceptable_threshold: float minimum probability required for
    positive classification
    :param display_graph: boolean displays graph if True
    """

    # Plot image:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(input_img, cmap='gray')

    # Add rectangles to image:
    for i in range(len(coords_list)):
        minr, minc, maxr, maxc = coords_list[i]  # y1, x1, width, height
        if flat_results[i][0] > acceptable_threshold:
            color = 'green'
        else:
            color = 'red'

        # Create Rectangles:
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor=color, linewidth=1,
                                  alpha=0.75)

        # Create label text:
        t = plt.text(minc+6, minr-8, 'flt: ' + '{0:.4f}'.format(
            flat_results[i][0]), fontsize=6)
        t.set_bbox(dict(facecolor=color, alpha=0.75, edgecolor=color))
        ax.add_patch(rect)

    plt.savefig(os.path.join(out_path, filename + '.png'))

    if display_graph:
        plt.show()
        ax.set_axis_off()
        plt.tight_layout()
