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
from itertools import combinations
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import spacer
from sklearn.metrics import mean_squared_error
from src.utils.aux_funcs import run_anova_test
from src.utils.aux_funcs import run_levene_test
from src.utils.aux_funcs import run_kruskal_test
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import run_mannwhitneyu_test
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import get_pearson_correlation
from src.utils.aux_funcs import add_confluence_group_col
from src.utils.aux_funcs import add_confluence_level_col
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

IOU_THRESHOLD = 0.3
DETECTION_THRESHOLD = 0.5
MASK_TYPE = 'ellipse'
PRINT_METRICS = True
PLOT_DATA = True
RUN_TESTS = True

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


def get_count_pairs_df(df: DataFrame) -> DataFrame:
    """
    Given a metrics df,
    returns count pairs df.
    """
    pass


def get_area_pairs_df(df: DataFrame) -> DataFrame:
    """
    Given a metrics df,
    returns area pairs df.
    """
    pass


def get_class_pairs_df(df: DataFrame) -> DataFrame:
    """
    Given a metrics df,
    returns class pairs df.
    """
    pass


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

    # printing spacer
    spacer()


def print_metrics_by_group(df: DataFrame,
                           group_col: str
                           ) -> None:
    """
    Given a metrics data frame,
    prints metrics by group.
    """
    # grouping df
    df_groups = df.groupby(group_col)

    # iterating over groups
    for df_name, df_group in df_groups:

        # printing group info
        group_string = f'Group: {df_name}'
        print(group_string)

        # printing global metrics
        print_metrics(df=df_group,
                      metrics_type='global')

        # printing mean metrics
        print_metrics(df=df_group,
                      metrics_type='mean')

        # printing spacer
        spacer()


def plot_f1_scores_by_col_lineplot(df: DataFrame,
                                   col_name: str,
                                   axis_name: str
                                   ) -> None:
    """
    Given a metrics data frame,
    plots F1-Score line plot.
    """
    # creating plot
    lineplot(data=df,
             x=col_name,
             y='f1_score',
             hue='cell_line',
             errorbar=None)

    # setting plot axis labels/limits
    plt.xlabel(axis_name)
    plt.ylabel('F1-Score')
    plt.ylim(0.0, 1.0)

    # adjusting layout
    plt.tight_layout()

    # showing plot
    plt.show()


def plot_f1_scores_by_col_boxplot(df: DataFrame,
                                  col_name: str,
                                  axis_name: str
                                  ) -> None:
    """
    Given a metrics data frame,
    plots F1-Score box plot.
    """
    # creating plot
    boxplot(data=df,
            x=col_name,
            y='f1_score')

    # setting plot axis labels/limits
    plt.xlabel(axis_name)
    plt.ylabel('F1-Score')
    plt.ylim(0.0, 1.0)

    # adjusting layout
    plt.tight_layout()

    # showing plot
    plt.show()


def run_cell_line_tests(df: DataFrame) -> None:
    """
    Given a metrics df, runs
    cell line tests.
    """
    # filtering df
    cols_to_keep = ['img_name',
                    'cell_line',
                    'fornma_count',
                    'f1_score']
    filtered_df = df[cols_to_keep]

    # defining group col
    group_col = 'cell_line'

    # grouping df by cell line
    df_groups = filtered_df.groupby(group_col)

    # defining placeholder value for anova samples
    samples = []

    # iterating over df groups
    for df_name, df_group in df_groups:

        # getting current sample count
        sample_count = len(df_group)

        # printing execution message
        f_string = f'cell line "{df_name}" count: {sample_count}'
        print(f_string)

        # getting current cell line f1 scores col
        current_sample_col = df_group['f1_score']

        # converting current sample to list
        current_sample = current_sample_col.to_list()

        # appending current sample to samples list
        samples.append(current_sample)

    # running Levene test
    run_levene_test(samples=samples)

    # running ANOVA test
    run_anova_test(samples=samples)

    # running Kruskal test
    run_kruskal_test(samples=samples)

    # getting sample pairs
    sample_pairs = combinations(iterable=samples,
                                r=2)

    # iterating over sample pairs
    for sample_pair in sample_pairs:

        # getting samples from pair
        sample_a, sample_b = sample_pair

        # running mann-whitney test
        run_mannwhitneyu_test(sample_a=sample_a,
                              sample_b=sample_b)


def run_count_tests(df: DataFrame) -> None:
    """
    Given a metrics df,
    runs nuclei count tests.
    """
    pass


def run_area_tests(df: DataFrame) -> None:
    """
    Given a metrics df,
    runs nuclei area tests.
    """
    pass


def run_erk_tests(df: DataFrame) -> None:
    """
    Given a metrics df, runs
    cit/nuc ratio tests.
    """
    print(df)
    print(df.columns)

    class_pairs_col = df['class_pairs']
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
    corr_value = run_anova_test(df=ratios_df,
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

    # TODO: remove once test completed
    metrics_df = metrics_df[metrics_df['fornma_count'] <= 200]

    # adding confluence group column
    add_confluence_group_col(df=metrics_df)

    # adding confluence level column
    add_confluence_level_col(df=metrics_df)

    # getting count pairs df
    count_pairs_df = get_count_pairs_df(df=metrics_df)

    # getting area pairs df
    area_pairs_df = get_area_pairs_df(df=metrics_df)

    # getting class pairs df
    class_pairs_df = get_class_pairs_df(df=metrics_df)

    # checking global bool
    if PRINT_METRICS:

        # printing global metrics
        print_metrics(df=metrics_df,
                      metrics_type='global')

        # printing mean metrics
        print_metrics(df=metrics_df,
                      metrics_type='mean')

        # printing metrics by cell line
        print_metrics_by_group(df=metrics_df,
                               group_col='cell_line')

        # printing metrics by confluence
        print_metrics_by_group(df=metrics_df,
                               group_col='confluence_group')

    # checking global bool
    if PLOT_DATA:

        # defining plots axis names
        confluence_axis_name = 'Confluence (%)'
        cell_line_axis_name = 'Cell line'

        # plotting F1-Scores by confluence line plot
        plot_f1_scores_by_col_lineplot(df=metrics_df,
                                       col_name='confluence_percentage_int',
                                       axis_name=confluence_axis_name)

        # plotting F1-Scores by confluence percentage box plot
        plot_f1_scores_by_col_boxplot(df=metrics_df,
                                      col_name='confluence_percentage_int',
                                      axis_name=confluence_axis_name)

        # plotting F1-Scores by confluence level box plot
        plot_f1_scores_by_col_boxplot(df=metrics_df,
                                      col_name='confluence_level',
                                      axis_name=confluence_axis_name)

        # plotting F1-Scores by cell line box plot
        plot_f1_scores_by_col_boxplot(df=metrics_df,
                                      col_name='cell_line',
                                      axis_name=cell_line_axis_name)

    # checking global bool
    if RUN_TESTS:

        # running cell line tests
        run_cell_line_tests(df=metrics_df)

        # running nuclei count tests
        run_count_tests(df=count_pairs_df)

        # running nuclei area tests
        run_area_tests(df=area_pairs_df)

        # running erk tests
        # run_erk_tests(df=class_pairs_df)

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
