# generate segmentation masks module

print('initializing...')  # noqa

# Code destined to analyse segmentation
# masks overlays based on BRAIND detections.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import imread
from cv2 import imwrite
from cv2 import putText
from cv2 import cvtColor
from numpy import arange
from os.path import join
from pandas import concat
from pandas import read_csv
from pandas import Series
from numpy import ndarray
from cv2 import contourArea
from cv2 import boundingRect
from cv2 import drawContours
from pandas import DataFrame
from cv2 import findContours
from cv2 import RETR_EXTERNAL
from cv2 import COLOR_GRAY2BGR
from cv2 import pointPolygonTest
from cv2 import CHAIN_APPROX_NONE
from numpy import uint8 as np_uint8
from argparse import ArgumentParser
from cv2 import FONT_HERSHEY_SIMPLEX
from src.utils.aux_funcs import draw_ellipse
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_segmentation_mask
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

######################################################################
# defining global variables

ER_MIN = 1.0
ER_MAX = 4.0
ER_STEP = 0.2

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'analyse segmentation masks overlays module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # detection file param
    parser.add_argument('-d', '--detection_file',
                        dest='detection_file',
                        required=True,
                        help='defines path to csv file containing model detections')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_single_er_df(df: DataFrame,
                     er: float
                     ) -> DataFrame:
    """
    Given a detections data frame,
    and an expansion ratio value,
    returns OBBs intersection df.
    """
    pass


def get_ers_df(df: DataFrame,
               er_min: float,
               er_max: float,
               er_step: float
               ) -> DataFrame:
    """
    Given a detections data frame,
    and parameters for expansion ratio
    range, returns masks overlays
    analysis data frame.
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # getting expansion ratio range
    ers_range = arange(start=er_min,
                       stop=er_max,
                       step=er_step)

    # getting expansion ratio list
    ers = [er for er in ers_range]
    ers_num = len(ers)

    # defining starter for current_er_index
    current_er_index = 1

    # iterating over er in er list
    for er in ers:

        # printing progress message
        base_string = f'calculating obbs intersection for er {er} | #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_er_index,
                               total=ers_num)

        # getting current er df
        current_er_df = get_single_er_df(df=df,
                                         er=er)

        # appending current er df to dfs list
        dfs_list.append(current_er_df)

        # updating current_er_index
        current_er_index += 1

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def analyse_segmentation_masks_overlays(detections_file: str,
                                        output_folder: str,
                                        expansion_ratio: float
                                        ) -> None:
    """
    Given paths to model detections file,
    creates analysis data frames and plots
    to assess overlay between OBBs in increasing
    expansion ratios.
    """
    # reading detections file
    print('reading detections file...')
    detections_df = read_csv(detections_file)

    # getting ers df
    ers_df = get_ers_df(df=detections_df,
                        er_min=ER_MIN,
                        er_max=ER_MAX,
                        er_step=ER_STEP)

    # printing execution message
    print(f'output saved to {output_folder}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting detections file
    detections_file = args_dict['detection_file']

    # getting output folder
    output_folder = args_dict['output_folder']

    # getting expansion ratio
    expansion_ratio = args_dict['expansion_ratio']
    expansion_ratio = float(expansion_ratio)

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_autophagy_dfs function
    analyse_segmentation_masks_overlays(detections_file=detections_file,
                                        output_folder=output_folder,
                                        expansion_ratio=expansion_ratio)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
