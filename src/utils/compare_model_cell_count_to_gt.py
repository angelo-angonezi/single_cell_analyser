# compare model cell count to ground-truth module

print('initializing...')  # noqa

# Code destined to comparing cell count data
# between model detections and gt annotations.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
import pandas as pd
from os.path import join
from pandas import read_csv
from pandas import DataFrame
from seaborn import lineplot
from argparse import ArgumentParser
from matplotlib import pyplot as plt
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

    # filtering df for fornmaVSmodels results only
    print('filtering df for fornmaVSmodels results only...')
    filtered_df = clean_df[clean_df['ann1'] == 'fornma']

    # adding F1-Scores to df
    print('adding F1-Scores to df...')
    f1_score_df = add_f1_score_column_to_df(df=filtered_df)

    # saving F1-Scores df
    print('saving F1-Scores df...')
    f1_score_df_save_path = join(output_folder,
                                 'f1_scores_df.csv')
    f1_score_df.to_csv(f1_score_df_save_path,
                       index=False)
    print('saved F1-Scores df in output folder.')

    # plotting precision-recall curves
    print('plotting precision-recall curves...')
    plot_prec_rec_curves(df=f1_score_df)

    # plotting F1-Score curve
    print('plotting F1-Score curve...')
    plot_f1_score_curve(df=f1_score_df)

    # printing execution message
    print('analysis complete.')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input file
    input_file = args_dict['input_file']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running analyse_compare_annotations_output function
    compare_model_cell_count_to_gt(input_file=input_file,
                                   output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
