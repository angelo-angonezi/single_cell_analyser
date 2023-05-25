# plot area histogram module

print('initializing...')  # noqa

# Code destined to plotting area
# histograms for model and annotator data.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from seaborn import histplot
from seaborn import FacetGrid
from seaborn import scatterplot
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import add_cell_area_col
from src.utils.aux_funcs import add_axis_ratio_col
from src.utils.aux_funcs import add_treatment_col_daph
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_merged_detection_annotation_df
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

DETECTION_THRESHOLD = 0.5

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "plot area histograms module"

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
                        required=True,
                        help=gt_help)

    # output folder param
    output_help = 'defines path to output folder'
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=False,
                        help=output_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_nma_df(df: DataFrame) -> DataFrame:
    """
    Given a merged detections/annotations data frame,
    returns a new df, of following structure:
    | evaluator | cell_area | axis_ratio |
    |   model   |   633.6   |   1.60913  |
    |   fornma  |   267.1   |   1.77106  |
    ...
    :param df: DataFrame. Represents merged detections/annotations data.
    :return: None.
    """
    # adding cell area column to df
    print('adding cell area column to df...')
    add_cell_area_col(df=df)

    # adding axis ratio column to df
    print('adding axis ratio column to df...')
    add_axis_ratio_col(df=df)

    # adding treatment column to df
    print('adding treatment column to df...')
    add_treatment_col_daph(df=df)

    # dropping unrequired cols
    all_cols = df.columns.to_list()
    keep_cols = ['cell_area', 'axis_ratio', 'evaluator', 'treatment']
    drop_cols = [col
                 for col
                 in all_cols
                 if col not in keep_cols]
    final_df = df.drop(drop_cols,
                       axis=1)

    # returning final df
    return final_df


def plot_histograms(df: DataFrame) -> None:
    """
    Given a nma data frame, plots cell_area and axis_ratio
    histograms, filtering df by control group.
    :param df: DataFrame. Represents NMA data.
    :return: None.
    """
    pass


def plot_area_histograms(detection_file_path: str,
                         ground_truth_file_path: str,
                         output_folder: float = 0.5
                         ) -> None:
    """
    Given paths to model detections and gt annotations,
    compares nma between evaluators, plotting
    comparison scatter plot.
    :param detection_file_path: String. Represents a file path.
    :param ground_truth_file_path: String. Represents a file path.
    :param output_folder: String. Represents detection threshold to be applied as filter.
    :return: None.
    """
    # getting merged detections df
    print('getting data from input files...')
    merged_df = get_merged_detection_annotation_df(detections_df_path=detection_file_path,
                                                   annotations_df_path=ground_truth_file_path)

    # filtering df by detection threshold
    print('filtering df by detection threshold...')
    filtered_df = merged_df[merged_df['detection_threshold'] >= DETECTION_THRESHOLD]

    # getting nma df
    nma_df = get_nma_df(df=filtered_df)

    # plotting histograms
    plot_histograms(df=nma_df)

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

    # running plot_area_histograms function
    plot_area_histograms(detection_file_path=detection_file,
                         ground_truth_file_path=ground_truth_file,
                         output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
