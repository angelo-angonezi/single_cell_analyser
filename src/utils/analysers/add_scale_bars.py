# add scale bars module
print('initializing...')  # noqa

# code destined to adding scale bars
# to annotation comparison images.

######################################################################
# imports

# importing required libraries

print('importing required libraries...')  # noqa
import cv2
import numpy as np
from os import listdir
from cv2 import imread
from cv2 import imwrite
from os.path import join
from src.utils.aux_funcs import print_progress_message
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global parameters

# defining file/folder paths
IMAGES_FOLDER_PATH = 'E:\\Angelo\\Desktop\\figs_paper_mestrado\\add_scale\\input'
OUTPUT_FOLDER_PATH = 'E:\\Angelo\\Desktop\\figs_paper_mestrado\\add_scale\\output'
SCALE_BAR_DICT = {'cell_count_outlier_img_ex_phase.png': 0.98,
                  'cell_count_outlier_img_ex_red.png': 0.98,
                  'bad_ex_01.png': 0.62,
                  'bad_ex_02.png': 0.62,
                  'good_ex_red.jpg': 0.62,
                  'good_ex_original.jpg': 0.62,
                  'good_ex_overlays.png': 0.62,
                  'confluence_11_example.png': 0.62,
                  'confluence_12_example.png': 0.62,
                  'A172_example_full.jpg': 0.62,
                  'A172_example.png': 0.62,
                  'MRC5_example_full.jpg': 0.62,
                  'MRC5_example.png': 0.62,
                  'U87_example_full.jpg': 0.62,
                  'U87_example.png': 0.62,
                  'U251_example_full.jpg': 0.62,
                  'U251_example.png': 0.62,
                  'outlier_01_full.png': 0.62,
                  'outlier_01.png': 0.62,
                  'outlier_02_full.png': 0.62,
                  'outlier_02.png': 0.62,
                  'outlier_03_full.png': 0.62,
                  'outlier_03.png': 0.62,
                  'contrast_challenge.png': 0.62,
                  'single_nuclei_01_phase.png': 0.62,
                  'single_nuclei_02_phase.png': 0.62,
                  'single_nuclei_03_phase.png': 0.62,
                  'single_nuclei_01_red.png': 0.62,
                  'single_nuclei_02_red.png': 0.62,
                  'single_nuclei_03_red.png': 0.62,
                  'alt_dataset_ex_01.png': 1.00,
                  'alt_dataset_ex_02.png': 0.75,
                  'alt_dataset_ex_03.png': 0.75,
                  'alt_dataset_ex_04.png': 0.62,
                  'alt_dataset_ex_05.png': 0.62,
                  'alt_dataset_ex_06.png': 0.62,
                  'iou_ex_a.png': 0.18,
                  'iou_ex_a2.png': 0.10,
                  'iou_ex_b.png': 0.12,
                  'iou_ex_b2.png': 0.10,
                  'iou_ex_c.png': 0.16,
                  'iou_ex_c2.png': 0.10,
                  'iou_ex_d.png': 0.17,
                  'iou_ex_d2.png': 0.10,
                  }

######################################################################
# defining auxiliary functions


def draw_rectangle(img,
                   cx: float,
                   cy: float,
                   width: float,
                   height: float,
                   angle: float,
                   color: tuple
                   ) -> None:
    """
    Given an open image, and corners for a rectangle with color,
    draws rectangle on base image.
    """
    # get the corner points
    box = cv2.boxPoints(((cx, cy),
                         (width, height),
                         angle))

    # converting corners format
    box = np.int0(box)

    # drawing lines
    cv2.drawContours(img, [box], -1, color, -1)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting images in input folder
    images = listdir(IMAGES_FOLDER_PATH)

    # defining global variables
    bar_width = 30  # in micrometers
    color = (255, 255, 255)

    # getting images num
    images_num = len(images)

    # defining starter for current image index
    current_image_index = 1

    # iterating over images in input folder
    for image in images:

        # printing execution message
        base_string = 'adding scale bar to image #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_image_index,
                               total=images_num)

        # getting image path
        image_path = join(IMAGES_FOLDER_PATH,
                          image)

        # reading current image
        open_image = imread(image_path)

        # getting current image resolution
        current_resolution = open_image.shape

        # getting current image width
        img_width, img_height, _ = current_resolution

        # defining bar height
        bar_height = 0.01 * img_height

        # getting current image scale bar
        current_scale_bar = SCALE_BAR_DICT[image]

        # getting current image pixel bar size (width)
        current_width = bar_width / current_scale_bar

        # defining coords
        cx = 0.02 * img_width
        cx = cx + (current_width / 2)
        cy = 0.02 * img_height

        # drawing rectangle
        draw_rectangle(img=open_image,
                       cx=cx,
                       cy=cy,
                       width=current_width,
                       height=bar_height,
                       angle=0,
                       color=color)

        # defining current save path
        save_path = join(OUTPUT_FOLDER_PATH,
                         image)

        # saving current image
        imwrite(save_path,
                open_image)

        # updating current image index
        current_image_index += 1

    # printing execution message
    print('scale bars added to all images.')

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
