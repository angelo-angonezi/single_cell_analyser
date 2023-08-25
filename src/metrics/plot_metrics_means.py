# plot metrics module
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
from pandas import read_csv
from pandas import DataFrame
from seaborn import lineplot
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
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

    # input path param
    parser.add_argument('-i', '--input-path',
                        dest='input_path',
                        required=True,
                        help='defines path to input (detection_metrics_df.csv) file')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output (.csv) file')

    # metric param
    parser.add_argument('-m', '--metric',
                        dest='metric',
                        required=False,
                        default='f1_mean',
                        help='defines metric to be plotted (precision_mean, recall_mean, f1_mean)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_metrics_means_df(df: DataFrame) -> DataFrame:
    """
    Given a metrics data frame, returns
    precision/recall/F1-Score means for
    all images.
    """
    # defining placeholder value for dfs_list
    dfs_list = []

    # grouping df
    groups_list = ['iou_threshold', 'detection_threshold', 'mask_style']
    df_groups = df.groupby(groups_list)
    groups_num = len(df_groups)

    # iterating over groups
    for group_index, group_info in enumerate(df_groups, 1):

        # getting group name/data
        group_name, group_data = group_info

        # getting current iou/dt
        iou, dt, mask_style = group_name

        # printing execution message
        progress_string = f'getting metrics mean for image #INDEX# of #TOTAL#'
        print_progress_message(base_string=progress_string,
                               index=group_index,
                               total=groups_num)

        # getting current group precision mean
        current_precision_col = group_data['precision']
        current_precision_mean = current_precision_col.mean()

        # getting current group recall mean
        current_recall_col = group_data['recall']
        current_recall_mean = current_recall_col.mean()

        # getting current group f1 mean
        current_f1_col = group_data['f1_score']
        current_f1_mean = current_f1_col.mean()

        # getting current group dict
        current_dict = {'iou_threshold': iou,
                        'detection_threshold': dt,
                        'mask_style': mask_style,
                        'precision_mean': current_precision_mean,
                        'recall_mean': current_recall_mean,
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


def plot_metric(input_path: str,
                output_path: str,
                metric: str
                ) -> None:
    # getting metrics df
    print('getting metrics df...')
    metrics_df = read_csv(input_path)

    # getting metrics means df
    print('getting metrics means df...')
    metrics_means_df = get_metrics_means_df(df=metrics_df)

    # filtering metrics means df by detection threshold
    # TODO: add detection threshold as execution parameter
    metrics_means_df = metrics_means_df[metrics_means_df['detection_threshold'] == 0.5]

    # saving metrics df
    metrics_means_df.to_csv(output_path)
    print(metrics_means_df)

    # plotting data
    print('plotting data...')
    lineplot(data=metrics_means_df,
             x='iou_threshold',
             y=metric,
             hue='mask_style')

    # showing plot
    plt.show()

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

    # getting input path
    input_path = args_dict['input_path']

    # getting output path
    output_path = args_dict['output_path']

    # getting metric
    metric = args_dict['metric']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running plot_metric function
    plot_metric(input_path=input_path,
                output_path=output_path,
                metric=metric)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
