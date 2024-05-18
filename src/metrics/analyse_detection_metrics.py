# analyse metrics module

print('initializing...')  # noqa

# Code destined to analysing data frame containing
# info on TP, FP, FN, precision, recall and F1-Score
# for each image in test set.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os import listdir
from numpy import arange
from pandas import concat
from seaborn import boxplot
from pandas import DataFrame
from seaborn import lineplot
from seaborn import stripplot
from pandas import read_pickle
from seaborn import scatterplot
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import get_pearson_correlation
from sklearn.metrics import mean_squared_error
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import add_confluence_group_col
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

IOU_THRESHOLD = 0.3
DETECTION_THRESHOLD = 0.5
MASK_TYPE = 'ellipse'

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'analyse detection metrics module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input path param
    parser.add_argument('-i', '--input-path',
                        dest='input_path',
                        required=True,
                        help='defines path to input (metrics_df.pickle) file')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to folder which will contain output files')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_global_metrics(df: DataFrame) -> tuple:
    """
    Given a metrics data frame,
    returns global metrics.
    """
    # getting tp/fp/fn columns
    tps_col = df['true_positives']
    fps_col = df['false_positives']
    fns_col = df['false_negatives']

    # getting tp/fp/fn values
    tps = tps_col.sum()
    fps = fps_col.sum()
    fns = fns_col.sum()

    # calculating precision/recall/f1
    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # assembling metrics tuple
    metrics = (precision, recall, f1_score)

    # returning metrics
    return metrics


def get_mean_metrics(df: DataFrame) -> tuple:
    """
    Given a metrics data frame,
    returns mean metrics.
    """
    # getting metrics columns
    precision_col = df['precision']
    recall_col = df['recall']
    f1_score_col = df['f1_score']

    # getting metric means
    precision = precision_col.mean()
    recall = recall_col.mean()
    f1_score = f1_score_col.mean()

    # assembling metrics tuple
    metrics = (precision, recall, f1_score)

    # returning metrics
    return metrics


def print_metrics(df: DataFrame,
                  metrics_type: str
                  ) -> None:
    """
    Given a metrics data frame and a metric
    type (global/mean), gets respective metrics
    and prints them on console.
    """
    # defining placeholder value for metrics
    precision = None
    recall = None
    f1_score = None

    # checking metrics type
    if metrics_type == 'global':

        # getting global metrics
        precision, recall, f1_score = get_global_metrics(df=df)

    # checking metrics type
    else:

        # getting mean metrics
        precision, recall, f1_score = get_mean_metrics(df=df)

    # rounding values before printing
    precision = round(precision, 2)
    recall = round(recall, 2)
    f1_score = round(f1_score, 2)

    # assembling metrics string
    metrics_string = f'--Detection Metrics ({metrics_type})--\n'
    metrics_string += f'Precision: {precision}\n'
    metrics_string += f'Recall: {recall}\n'
    metrics_string += f'F1-Score: {f1_score}'

    # printing metrics string
    print(metrics_string)


def print_metrics_by_group(df: DataFrame) -> None:
    """
    Given a metrics data frame,
    prints metrics by group.
    """
    # defining groups
    groups = 'cell_line'

    # grouping df
    df_groups = df.groupby(groups)

    # iterating over groups
    for df_name, df_group in df_groups:

        # printing group info
        group_string = f'Group: {df_name}'
        print(group_string)

        # printing global metrics
        # print_metrics(df=df_group,
        #               metrics_type='global')

        # printing mean metrics
        print_metrics(df=df_group,
                      metrics_type='mean')
        spacer()


def plot_f1_scores_lineplot(df: DataFrame) -> None:
    """
    Given a metrics data frame,
    plots F1-Score line plot.
    """
    # creating plot
    lineplot(data=df,
             x='confluence_percentage_int',
             y='f1_score',
             hue='cell_line',
             errorbar=None)

    # setting plot axis labels/limits
    plt.ylabel('F1-Score')
    plt.ylim(0.0, 1.0)

    # adjusting layout
    plt.tight_layout()

    # showing plot
    plt.show()


def plot_f1_scores_boxplot(df: DataFrame) -> None:
    """
    Given a metrics data frame,
    plots F1-Score box plot.
    """
    # creating plot
    boxplot(data=df,
            x='confluence_percentage_int',
            y='f1_score')
    # stripplot(data=df,
    #           x='confluence_percentage_int',
    #           y='f1_score')

    # setting plot axis labels/limits
    plt.ylabel('F1-Score')
    plt.ylim(0.0, 1.0)

    # adjusting layout
    plt.tight_layout()

    # showing plot
    plt.show()


def analyse_metrics(input_path: str,
                    output_folder: str,
                    ) -> None:
    # getting metrics df
    print('getting metrics df...')
    metrics_df = read_pickle(input_path)

    # filtering df
    metrics_df = metrics_df[metrics_df['detection_threshold'] == DETECTION_THRESHOLD]
    metrics_df = metrics_df[metrics_df['iou_threshold'] == IOU_THRESHOLD]
    metrics_df = metrics_df[metrics_df['mask_style'] == MASK_TYPE]

    # adding confluence column
    add_confluence_group_col(df=metrics_df)

    # printing global metrics
    print_metrics(df=metrics_df,
                  metrics_type='global')
    spacer()

    # printing mean metrics
    print_metrics(df=metrics_df,
                  metrics_type='mean')
    spacer()

    # printing metrics by groups
    print_metrics_by_group(df=metrics_df)

    # plotting F1-Scores line plot
    # plot_f1_scores_lineplot(df=metrics_df)

    # plotting F1-Scores box plot
    # plot_f1_scores_boxplot(df=metrics_df)

    print(metrics_df)
    print(metrics_df.columns)

    class_pairs_col = metrics_df['class_pairs']
    class_pairs_list = class_pairs_col.to_list()
    braind_ratios = []
    fornma_ratios = []
    for i in class_pairs_list:
        for j in i:
            braind_ratio, fornma_ratio = j
            braind_ratios.append(braind_ratio)
            fornma_ratios.append(fornma_ratio)

    ratios_dict = {'braind_ratio': braind_ratios,
                   'fornma_ratio': fornma_ratios}
    ratios_df = DataFrame(ratios_dict)
    print(ratios_df)
    scatterplot(data=ratios_df,
                x='fornma_ratio',
                y='braind_ratio')
    corr_value = get_pearson_correlation(df=ratios_df,
                                         col_real='fornma_ratio',
                                         col_pred='braind_ratio')
    corr_value = round(corr_value, 3)

    ratios_df = ratios_df.melt()
    print(ratios_df)
    from seaborn import histplot
    histplot(data=ratios_df,
             x='value',
             hue='variable')
    plt.show()
    title = f'ERK-KTR Histogram Plot'
    plt.title(title)
    plt.show()
    exit()

    title = f'ERK-KTR Correlation Plot | Pearson R: {corr_value}'
    plt.title(title)
    # plt.show()

    ratios_df['ratio_error'] = ratios_df['braind_ratio'] - ratios_df['fornma_ratio']
    ratios_df['ratio_error_abs'] = [abs(v) for v in ratios_df['ratio_error']]
    print(ratios_df)
    mae = ratios_df['ratio_error_abs'].mean()
    print(mae)
    print(len(ratios_df))

    # printing execution message
    print(f'output saved to "{output_folder}".')
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
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running plot_metric function
    analyse_metrics(input_path=input_path,
                    output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
