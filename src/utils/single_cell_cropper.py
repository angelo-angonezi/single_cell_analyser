# single cell cropper module

print('initializing...')  # noqa

# code destined to cropping single cells
# using ML output.

######################################################################
# importing required libraries

print('importing required libraries...')  # noqa
import cv2
from time import sleep
from os.path import join
from numpy import ndarray
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import flush_or_print
from src.utils.aux_funcs import get_obbs_from_df
from src.utils.aux_funcs import get_data_from_consolidated_df
print('all required libraries successfully imported.')  # noqa
sleep(0.8)

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "single cell cropper - tool used to segment cells based on\n"
    description += "machine learning output data.\n"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser
    parser.add_argument('-i', '--images-input-folder',
                        dest='images_input_folder',
                        help='defines path to folder containing images',
                        required=True)

    parser.add_argument('-d', '--detections-dataframe',
                        dest='detections_df_path',
                        help='defines path to file containing detections info',
                        required=True)

    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        help='defines path to output folder which will contain crops',
                        required=True)

    parser.add_argument('-r', '--resize',
                        dest='resize_toggle',
                        action='store_true',
                        help='resizes all crops to same dimensions',
                        default=False,
                        required=False)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def crop_single_obb(image: ndarray,
                    obb: tuple
                    ) -> ndarray:
    """
    Given an array representing an image,
    and a tuple containing obb's info,
    returns given obb crop, rotated to
    be aligned to x-axis.
    :param image: Array. Represents an open image.
    :param obb: Tuple. Represents obb's info.
    :return: Array. Represents obb crop from image.
    """
    pass


def crop_multiple_obbs(image: ndarray,
                       obbs_list: list,
                       output_folder: str
                       ) -> None:
    """
    Given an array representing an image,
    and a list of tuples containing obb's
    info, crops obbs in current image,
    saving crops to given output folder.
    :param image: Array. Represents an open image.
    :param obbs_list: List. Represents a list of obbs.
    :param output_folder: String. Represents a path to a folder.
    :return: None.
    """
    pass


def get_single_image_crops(image: ndarray,
                           image_group: DataFrame,
                           output_folder: str
                           ) -> None:
    """
    Given an array representing an image,
    and a data frame representing current
    image obbs detections, saves crops
    of obbs in given output folder.
    :param image: Array. Represents an open image.
    :param image_group: DataFrame. Represents current image obbs.
    :param output_folder: String. Represents a path to a folder.
    :return: None.
    """
    pass


def get_multiple_image_crops(consolidated_df: DataFrame,
                             input_folder: str,
                             output_folder: str
                             ) -> None:
    """
    Given ML detections consolidated data frame,
    a path to an input folder containing images,
    saves obbs crops in output folder.
    :param consolidated_df: DataFrame. Represents ML obbs
    detections for images in input folder.
    :param input_folder: String. Represents a path to a folder.
    :param output_folder: String. Represents a path to a folder.
    :return: None.
    """
    # getting data from consolidated df csv
    print('getting data from consolidated df...')
    main_df = get_data_from_consolidated_df(consolidated_df_file_path=con)
    pass


def single_cell_cropper(input_folder: str,
                        detections_df_path: str,
                        output_folder: str,
                        resize_toggle: bool
                        ) -> None:
    """
    Given execution parameters, runs
    cropping function on multiple images.
    :param input_folder: String. Represents a path to a folder.
    :param detections_df_path: String. Represents a path to a file.
    :param output_folder: String. Represents a path to a folder.
    :param resize_toggle: Bool. Represents a toogle.
    :return: None.
    """
    pass

######################################################################
# defining main function


def main():
    """
    Runs main code.
    """
    # getting data from Argument Parser
    args_dict = get_args_dict()

    # getting input folder
    input_folder = args_dict['images_input_folder']

    # getting detections df path
    detections_df_path = args_dict['detections_df_path']

    # getting output folder
    output_folder = args_dict['output_folder']

    # getting resize toggle
    resize_toggle = args_dict['resize_toggle']

    # running single cell cropper function
    single_cell_cropper(input_folder=input_folder,
                        detections_df_path=detections_df_path,
                        output_folder=output_folder,
                        resize_toggle=resize_toggle)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
