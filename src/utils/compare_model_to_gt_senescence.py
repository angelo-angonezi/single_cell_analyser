# compare model senescence to ground-truth module

print('initializing...')  # noqa

# Code destined to comparing senescence results
# between model detections and gt annotations.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
import pandas as pd
from pandas import DataFrame
from seaborn import histplot
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import add_cell_area_col
from src.utils.aux_funcs import add_treatment_col_debs
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_merged_detection_annotation_df
print('all required libraries successfully imported.')  # noqa

# next line prevents "SettingWithCopyWarning" pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

#####################################################################
# defining global variables

GRAY_AREA_MIN_STD = 1.7
GRAY_AREA_MAX_STD = 3.0

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "compare model senescence to gt module"

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

    # dt file param
    dt_help = 'defines detection threshold to be used as filter for model detections'
    parser.add_argument('-t', '--detection-threshold',
                        dest='detection_threshold',
                        required=False,
                        help=dt_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_gray_area_cutoff_values(df: DataFrame,
                                evaluator: str
                                ) -> tuple:
    """
    Given a merged detections/annotations data frame,
    and an evaluator name, returns gray area cutoff
    values for current evaluator.
    OBS: Takes only CTR treatment group into consideration.
    :param df: DataFrame. Represents merged detections/annotations data.
    :param evaluator: str. Represents an evaluator name ('model'/'fornma').
    :return: Tuple. Represents min and max cutoff values for gray area.
    """
    # filtering df by evaluator
    df = df[df['evaluator'] == evaluator]

    # getting control group df
    control_df = df[df['treatment'] == 'CTR']

    # getting current evaluator mean/std area values
    area_mean = control_df['cell_area'].mean()
    area_std = control_df['cell_area'].std()

    # defining cutoff values
    gray_area_min = area_mean + (GRAY_AREA_MIN_STD * area_std)
    gray_area_max = area_mean + (GRAY_AREA_MAX_STD * area_std)

    # assembling cutoff_tuple
    cutoff_tuple = (gray_area_min, gray_area_max)

    # returning cutoff_tuple
    return cutoff_tuple


def add_cell_size_col(df: DataFrame,
                      fornma_cutoff_values: tuple,
                      model_cutoff_values: tuple
                      ) -> None:
    """
    Given a merged detections/annotations data frame,
    adds 'cell_size' column, obtained by analysing
    each evaluator group mean/std area values and
    comparing them to given cutoff values.
    :param df: DataFrame. Represents merged detections/annotations data.
    :param fornma_cutoff_values: Tuple. Represents cell_area cutoff values.
    :param model_cutoff_values: Tuple. Represents cell_area cutoff values.
    :return: None.
    """
    # adding senescence placeholder column to df
    df['cell_size'] = None

    # getting current evaluator df rows
    df_rows = df.iterrows()

    # iterating over df rows
    for row_index, row_data in df_rows:

        # getting current cell evaluator
        current_cell_evaluator = row_data['evaluator']

        # defining current cell cutoffs
        gray_area_tuple = fornma_cutoff_values if current_cell_evaluator == 'fornma' else model_cutoff_values
        gray_area_min, gray_area_max = gray_area_tuple

        # getting current cell area value
        current_cell_area = row_data['cell_area']

        # defining current cell size
        current_cell_size = 'Small'
        if current_cell_area > gray_area_min:
            current_cell_size = 'Medium'
        if current_cell_area > gray_area_max:
            current_cell_size = 'Large'

        # updating current line cell_size value
        df.at[row_index, 'cell_size'] = current_cell_size


def get_cell_size_df(df: DataFrame) -> DataFrame:
    """
    Given a merged detections/annotations data frame,
    returns a new df, of following structure:
    | evaluator | cell_area | axis_ratio | treatment | cell_size |
    |   model   |   633.6   |   1.60913  |    TMZ    |   Small   |
    |   fornma  |   267.1   |   1.77106  |    CTR    |  Medium   |
    |   fornma  |   328.9   |   1.58120  |    CTR    |   Large   |
    ...
    :param df: DataFrame. Represents merged detections/annotations data.
    :return: DataFrame. Represents a cell size data frame.
    """
    # adding cell area column to df
    print('adding cell area column to df...')
    add_cell_area_col(df=df)

    # adding treatment column to df
    print('adding treatment column to df...')
    add_treatment_col_debs(df=df)

    # getting evaluators gray area cutoff values
    fornma_cutoff_values = get_gray_area_cutoff_values(df=df,
                                                       evaluator='fornma')
    model_cutoff_values = get_gray_area_cutoff_values(df=df,
                                                      evaluator='model')

    # printing execution message
    f_string = f'fornma gray area min/max cutoffs: {fornma_cutoff_values}\n'
    f_string += f'model gray area min/max cutoffs: {model_cutoff_values}'
    print(f_string)

    # adding cell size column to df
    print('adding cell size column to df...')
    add_cell_size_col(df=df,
                      fornma_cutoff_values=fornma_cutoff_values,
                      model_cutoff_values=model_cutoff_values)

    # dropping unrequired cols
    all_cols = df.columns.to_list()
    keep_cols = ['cell_area', 'evaluator', 'treatment', 'cell_size']
    drop_cols = [col
                 for col
                 in all_cols
                 if col not in keep_cols]
    final_df = df.drop(drop_cols,
                       axis=1)

    # returning final df
    return final_df


def plot_cell_area_histograms(df: DataFrame) -> None:
    """
    Given a senescence data frame, plots cell_area and axis_ratio
    histograms, filtering df by control group.
    :param df: DataFrame. Represents senescence data.
    :return: None.
    """
    # grouping df by evaluator
    evaluator_groups = df.groupby('evaluator')

    # iterating over evaluator groups
    for evaluator_name, evaluator_group in evaluator_groups:

        # creating histogram
        histplot(data=evaluator_group,
                 x='cell_area',
                 hue='treatment',
                 kde=True)

        # setting plot title
        title = f'Cell area histogram ({evaluator_name})'
        plt.title(title)

        # showing plot
        plt.show()

        # closing plot
        plt.close()


def plot_senescence_data(df: DataFrame) -> None:
    """
    Given a senescence data frame, plots senescence data,
    coloring data by evaluator (model detections and
    fornma annotations).
    :param df: DataFrame. Represents senescence data.
    :return: None.
    """
    pass

    # showing plot
    plt.show()


def compare_model_to_gt_senescence(detection_file_path: str,
                                   ground_truth_file_path: str,
                                   detection_threshold: float
                                   ) -> None:
    """
    Given paths to model detections and gt annotations,
    compares cell count between evaluators, plotting
    comparison scatter plot.
    :param detection_file_path: String. Represents a file path.
    :param ground_truth_file_path: String. Represents a file path.
    :param detection_threshold: Float. Represents detection threshold to be applied as filter.
    :return: None.
    """
    # getting merged detections df
    print('getting data from input files...')
    merged_df = get_merged_detection_annotation_df(detections_df_path=detection_file_path,
                                                   annotations_df_path=ground_truth_file_path)

    # filtering df by detection threshold
    print('filtering df by detection threshold...')
    filtered_df = merged_df[merged_df['detection_threshold'] >= detection_threshold]

    # getting cell size data
    print('getting cell size df...')
    cell_size_df = get_cell_size_df(df=filtered_df)

    # plotting senescence data
    print('plotting cell area histograms...')
    plot_cell_area_histograms(df=cell_size_df)

    # plotting senescence data
    print('plotting senescence data...')
    plot_senescence_data(df=cell_size_df)

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

    # getting detection threshold
    detection_threshold = args_dict['detection_threshold']
    detection_threshold = float(detection_threshold)

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running compare_model_to_gt_senescence function
    compare_model_to_gt_senescence(detection_file_path=detection_file,
                                   ground_truth_file_path=ground_truth_file,
                                   detection_threshold=detection_threshold)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
