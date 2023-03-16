# compare model cell count to ground-truth module

print('initializing...')  # noqa

# Code destined to comparing cell count data
# between model detections and gt annotations.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
import pandas as pd
from pandas import concat
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_merged_detection_annotation_df
print('all required libraries successfully imported.')  # noqa

# next line prevents "SettingWithCopyWarning" pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "analyse compare_annotations.py output module"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # detection file param
    detection_help = 'defines path to csv file containing model detections'
    parser.add_argument('-d', '--detection_file',
                        dest='detection_file',
                        required=True,
                        help=detection_help)

    # gt file param
    gt_help = 'defines path to csv file containing ground-truth annotations'
    parser.add_argument('-g', '--ground-truth-file',
                        dest='ground_truth_file',
                        required=False,
                        help=gt_help)

    # output folder param
    output_help = 'defines output folder (folder that will contain output csvs/pngs)'
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


def get_cell_count_df(df: DataFrame) -> DataFrame:
    """
    Given a merged detections/annotations data frame,
    returns cell count data frame, of following structure:
    | img_name | evaluator | cell_count |
    | img1.png |   model   |     62     |
    | img1.png |  fornma   |     58     |
    ...
    :param df: DataFrame. Represents merged detections/annotations data.
    :return: DataFrame. Represents cell count data frame.
    """
    # defining placeholder value for dfs_list
    dfs_list = []

    # grouping df by
    print(df)



    exit()


def compare_model_cell_count_to_gt(detection_file_path: str,
                                   ground_truth_file_path: str,
                                   output_folder: str
                                   ) -> None:
    """
    Given an input file (compare_annotations.py output),
    generates precision-recall curves, and saves analysis
    dfs in given output folder.
    :param detection_file_path: String. Represents a file path.
    :param ground_truth_file_path: String. Represents a file path.
    :param output_folder: String. Represents a path to a folder.
    :return: None.
    """
    # getting merged detections df
    merged_df = get_merged_detection_annotation_df(detections_df_path=detection_file_path,
                                                   annotations_df_path=ground_truth_file_path)

    # getting ready-to-plot data


    # printing execution message
    print('analysis complete.')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting detection file path
    detection_file = args_dict['detection_file']

    # getting ground-truth file path
    ground_truth_file = args_dict['ground_truth_file']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running compare_model_cell_count_to_gt function
    compare_model_cell_count_to_gt(detection_file_path=detection_file,
                                   ground_truth_file_path=ground_truth_file,
                                   output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
