# generate autophagy df module

print('initializing...')  # noqa

# Code destined to generating autophagy
# data frame, based on segmentation masks.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import imread
from os.path import join
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from cv2 import findContours
from cv2 import RETR_EXTERNAL
from cv2 import CHAIN_APPROX_NONE
from argparse import ArgumentParser
from src.utils.aux_funcs import get_crop_pixels
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

######################################################################
# defining global variables

MIN_CELL_AREA = 2
MIN_FOCI_AREA = 2

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'generate autophagy df module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # cell masks folder param
    parser.add_argument('-c', '--cell-masks-folder',
                        dest='cell_masks_folder',
                        required=True,
                        help='defines path to folder containing cell segmentation masks (macro output)')

    # foci masks folder param
    parser.add_argument('-f', '--foci-masks-folder',
                        dest='foci_masks_folder',
                        required=True,
                        help='defines path to folder containing foci segmentation masks (macro output)')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output file (.csv)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def generate_autophagy_df(cell_masks_folder: str,
                          foci_masks_folder: str,
                          output_path: str,
                          ) -> None:
    """
    Given paths to folders containing segmentation
    masks (cells/foci), analyses images to generate
    an autophagy analysis data frame.
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # getting images input folder
    images_list = get_specific_files_in_folder(path_to_folder=cell_masks_folder,
                                               extension='.tif')

    # iterating over images list
    for image_name in images_list:

        # getting image paths
        cell_masks_path = join(cell_masks_folder,
                               image_name)
        foci_masks_path = join(foci_masks_folder,
                               image_name)

        # reading images
        cell_masks_img = imread(cell_masks_path,
                                -1)
        foci_masks_img = imread(foci_masks_path,
                                -1)

        # getting image contours
        cell_masks_contours, _ = findContours(cell_masks_img, RETR_EXTERNAL, CHAIN_APPROX_NONE)
        foci_masks_contours, _ = findContours(foci_masks_img, RETR_EXTERNAL, CHAIN_APPROX_NONE)

        # filtering contours by area
        # TODO: add this part

        # drawing contours
        import cv2
        cell_masks_img_rgb = cv2.cvtColor(cell_masks_img, cv2.COLOR_GRAY2BGR)
        save_path = output_path.replace('.csv', '_cell.png')
        cv2.drawContours(cell_masks_img_rgb, cell_masks_contours, -1, (0, 255, 0), 1)
        cv2.imwrite(save_path,
                    cell_masks_img_rgb)

        foci_masks_img_rgb = cv2.cvtColor(foci_masks_img, cv2.COLOR_GRAY2BGR)
        save_path = output_path.replace('.csv', '_foci.png')
        cv2.drawContours(foci_masks_img_rgb, foci_masks_contours, -1, (0, 255, 0), 1)
        cv2.imwrite(save_path,
                    foci_masks_img_rgb)

    # printing execution message
    print(f'output saved to {output_path}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting cell folder path
    cell_masks_folder = args_dict['cell_masks_folder']

    # getting foci folder path
    foci_masks_folder = args_dict['foci_masks_folder']

    # getting output path
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running generate_autophagy_df function
    generate_autophagy_df(cell_masks_folder=cell_masks_folder,
                          foci_masks_folder=foci_masks_folder,
                          output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
