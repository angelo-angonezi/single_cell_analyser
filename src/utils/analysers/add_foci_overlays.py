# add foci overlay module

print('initializing...')  # noqa

# Code destined to generating images with foci
# overlays based on model detections (and GT annotations).

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import imread
from os import environ
from cv2 import imwrite
from cv2 import putText
from cv2 import cvtColor
from os.path import join
from numpy import ndarray
from pandas import Series
from os.path import exists
from pandas import DataFrame
from cv2 import drawContours
from cv2 import findContours
from cv2 import COLOR_BGR2RGB
from cv2 import COLOR_RGB2BGR
from cv2 import RETR_EXTERNAL
from cv2 import COLOR_GRAY2BGR
from cv2 import pointPolygonTest
from cv2 import CHAIN_APPROX_NONE
from argparse import ArgumentParser
from cv2 import FONT_HERSHEY_SIMPLEX
from src.utils.aux_funcs import draw_circle
from src.utils.aux_funcs import load_bgr_img
from src.utils.aux_funcs import draw_ellipse
from src.utils.aux_funcs import draw_rectangle
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import load_grayscale_img
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
from src.utils.aux_funcs import get_merged_detection_annotation_df
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "add foci overlays to images based on model detections (and GT annotations)"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help='defines path to folder containing fluorescent crops')

    # foci masks folder param
    parser.add_argument('-f', '--foci-masks-folder',
                        dest='foci_masks_folder',
                        required=True,
                        help='defines path to folder containing foci segmentation masks')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # output folder param
    output_help = 'defines output folder (folder that will contain outlined images)'
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help=output_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def add_overlays_to_single_image(image_path: str,
                                 mask_path: str,
                                 output_path: str
                                 ) -> None:
    """
    Given paths to an image and respective
    segmentation masks, adds foci overlays,
    saving results to given output path.
    """
    # opening image/mask
    image = load_bgr_img(image_path=image_path)
    mask = load_grayscale_img(image_path=mask_path)

    # finding mask contours
    contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_NONE)

    # defining contour parameters
    thickness = 1
    color = (0, 255, 0)

    # drawing contours on image
    drawContours(image,
                 contours,
                 -1,
                 color,
                 thickness)

    # saving image
    imwrite(output_path,
            image)


def add_overlays_to_multiple_images(input_folder: str,
                                    images_extension: str,
                                    foci_masks_folder: str,
                                    output_folder: str
                                    ) -> None:
    """
    Given paths to folders containing original
    images and respective detected foci masks,
    adds foci overlays to original images,
    saving results to given output folder.
    """
    # getting images in input folder
    images = get_specific_files_in_folder(path_to_folder=input_folder,
                                          extension=images_extension)
    images_num = len(images)

    # defining placeholder value for current image index
    current_image_index = 1

    # iterating over images
    for image in images:

        # printing execution message
        progress_base_string = f'adding overlays to image #INDEX# of #TOTAL#'
        print_progress_message(base_string=progress_base_string,
                               index=current_image_index,
                               total=images_num)

        # updating current image index
        current_image_index += 1

        # getting current image/mask/output paths
        image_path = join(input_folder,
                          image)
        mask_path = join(foci_masks_folder,
                         image)
        output_path = join(output_folder,
                           image)

        # adding overlays to current image
        add_overlays_to_single_image(image_path=image_path,
                                     mask_path=mask_path,
                                     output_path=output_path)

    # printing execution message
    f_string = f'overlays added to all {images_num} images!'
    print(f_string)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input folder
    input_folder = args_dict['input_folder']

    # getting foci folder path
    foci_masks_folder = args_dict['foci_masks_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running add_overlays_to_multiple_images function
    add_overlays_to_multiple_images(input_folder=input_folder,
                                    foci_masks_folder=foci_masks_folder,
                                    images_extension=images_extension,
                                    output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
