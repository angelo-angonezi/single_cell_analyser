# compare model cell count to ground-truth module
import matplotlib.pyplot as plt

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
from seaborn import scatterplot
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

    # grouping df
    df_groups = df.groupby(['img_file_name', 'evaluator'])

    # iterating over df groups
    for df_name, df_group in df_groups:

        # getting current df img_file_name and evaluator
        current_img, current_evaluator = df_name

        # getting current group cell count
        current_cell_count = len(df_group)

        # assembling current group dict
        current_group_dict = {'img_name': current_img,
                              'evaluator': current_evaluator,
                              'cell_count': current_cell_count}

        # assembling current group df
        current_group_df = DataFrame(current_group_dict,
                                     index=[0])

        # appending current group df to dfs_list
        dfs_list.append(current_group_df)

    # concatenating dfs in dfs_list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final_df
    return final_df


def plot_cell_count_data(df: DataFrame) -> None:
    """
    Given a cell count data frame,
    plots scatter plot to compare
    detectionsVSannotations.
    :param df: DataFrame. Represents cell count data frame.
    :return: None.
    """
    # plotting data
    scatterplot(data=df,
                x='img_name',
                y='cell_count',
                hue='evaluator')

    # setting title/axis names
    plt.title('Cell count comparison')
    plt.ylabel('Cell count')
    plt.xlabel('Image name')
    plt.xticks(rotation=15)

    # adding legend
    plt.legend(title='Evaluator',
               loc='upper right')

    # showing plot
    plt.show()
    # plt.savefig('E:\Angelo\Desktop\ml_temp\output.png')


def compare_model_cell_count_to_gt(detection_file_path: str,
                                   ground_truth_file_path: str,
                                   ) -> None:
    """
    Given paths to model detections and gt annotations,
    compares cell count between evaluators, plotting
    comparison scatter plot.
    :param detection_file_path: String. Represents a file path.
    :param ground_truth_file_path: String. Represents a file path.
    :return: None.
    """
    # getting merged detections df
    print('getting data from input files...')
    merged_df = get_merged_detection_annotation_df(detections_df_path=detection_file_path,
                                                   annotations_df_path=ground_truth_file_path)

    # getting cell count data
    print('getting cell count df...')
    cell_count_df = get_cell_count_df(df=merged_df)

    # plotting cell count data
    print('plotting cell count data...')
    plot_cell_count_data(df=cell_count_df)

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

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running compare_model_cell_count_to_gt function
    compare_model_cell_count_to_gt(detection_file_path=detection_file,
                                   ground_truth_file_path=ground_truth_file)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
