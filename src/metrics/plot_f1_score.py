# plot F1-Score module
import matplotlib.pyplot as plt

print('initializing...')  # noqa

# Code destined to generating data frame containing
# info on TP, FP, FN, precision, recall and F1-Score
# for each image in test set.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from numpy import arange
from pandas import concat
from pandas import Series
from numpy import ndarray
from pandas import read_csv
from pandas import DataFrame
from seaborn import lineplot
from numpy import add as np_add
from numpy import count_nonzero
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from numpy import zeros as np_zeroes
from src.utils.aux_funcs import draw_circle
from src.utils.aux_funcs import draw_ellipse
from src.utils.aux_funcs import draw_rectangle
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import simple_hungarian_algorithm
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_merged_detection_annotation_df
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

START = 0.0
STOP = 1.0
IOU_STEP = 0.05
DETECTION_STEP = 0.1
IOU_RANGE = arange(START,
                   STOP + IOU_STEP,
                   IOU_STEP)
DETECTION_RANGE = arange(START,
                         STOP + DETECTION_STEP,
                         DETECTION_STEP)
IOU_THRESHOLDS = [round(i, 2) for i in IOU_RANGE]
DETECTION_THRESHOLDS = [round(i, 2) for i in DETECTION_RANGE]

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'plot F1-Score module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # fornma file param
    parser.add_argument('-f', '--fornma-file',
                        dest='fornma_file',
                        required=True,
                        help='defines path to fornma (.csv) file')

    # detections file param
    parser.add_argument('-d', '--detections-file',
                        dest='detections_file',
                        required=True,
                        help='defines path to detections (.csv) file')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output (.csv) file')

    # style param
    parser.add_argument('-s', '--mask-style',
                        dest='mask_style',
                        required=False,
                        default='ellipse',
                        help='defines overlay style (rectangle/circle/ellipse)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_f1_means_df(df: DataFrame) -> DataFrame:
    """
    Given a metrics data frame, returns
    F1-Score means for all images.
    """
    # defining placeholder value for dfs_list
    dfs_list = []

    # grouping df
    groups_list = ['iou_threshold', 'detection_threshold']
    df_groups = df.groupby(groups_list)
    groups_num = len(df_groups)

    # iterating over groups
    for group_index, group_info in enumerate(df_groups, 1):

        # getting group name/data
        group_name, group_data = group_info

        # getting current iou/dt
        iou, dt = group_name

        # printing execution message
        progress_string = f'getting F1-Score mean for image #INDEX# of #TOTAL#'
        print_progress_message(base_string=progress_string,
                               index=group_index,
                               total=groups_num)

        # getting current group f1 mean
        current_f1_col = group_data['f1_score']
        current_f1_mean = current_f1_col.mean()

        # getting current group dict
        current_dict = {'iou_threshold': iou,
                        'detection_threshold': dt,
                        'f1_mean': current_f1_mean}

        # getting current group df
        current_df = DataFrame(current_dict,
                               index=[0])

        # appending current df to dfs_list
        dfs_list.append(current_df)

    # concatenating dfs in dfs_list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def plot_f1_score(input_path: str,
                  output_path: str,
                  ) -> None:
    # getting metrics df
    print('getting metrics df...')
    metrics_df = read_csv(input_path)

    # getting f1 score means df
    print('getting f1 score means df...')
    f1_means_df = get_f1_means_df(df=metrics_df)

    # plotting data
    print('plotting data...')
    lineplot(data=f1_means_df,
             x='iou_threshold',
             y='f1_mean',
             hue='detection_threshold')

    # showing plot
    plt.show()

    # saving plot
    print('saving plot...')
    # plt.savefig(output_path)

    # printing execution message
    print(f'output saved to "{output_path}"')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """
    Gets execution parameters from
    command line and runs main function.
    """
    # getting args dict
    args_dict = get_args_dict()

    # getting fornma file
    fornma_file = args_dict['fornma_file']

    # getting detections file
    detections_file = args_dict['detections_file']

    # getting output path
    output_path = args_dict['output_path']

    # getting mask style
    mask_style = args_dict['mask_style']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running plot_f1_score function
    plot_f1_score(fornma_file=fornma_file,
                                  detections_file=detections_file,
                                  output_path=output_path,
                                  iou_thresholds=IOU_THRESHOLDS,
                                  detection_thresholds=DETECTION_THRESHOLDS,
                                  style=mask_style)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
