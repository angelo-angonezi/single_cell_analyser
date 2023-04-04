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
from seaborn import regplot
from pandas import DataFrame
from seaborn import scatterplot
from scipy.stats import linregress
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


def plot_linear_regression(df: DataFrame,
                           x_col: str,
                           y_col: str
                           ) -> None:
    """
    Given a df, calculates linear regression for given x and y cols.
    :param df: DataFrame. Represents esiCancer joined output file (with pop groups col added).
    :param x_col: String. Represents a data frame column name.
    :param y_col: String. Represents a data frame column name.
    :return:
    """
    # getting xy data from current group
    x_data = df[x_col]
    y_data = df[y_col]

    # calculating linear regression with xy data
    linreg = linregress(x_data, y_data)

    # splitting results' coefficients
    slope, intercept, r_value, p_value, std_err = linreg

    # rounding values for plot
    round_slope = round(slope, 5)
    round_intercept = round(intercept, 5)
    round_r_value = round(r_value, 2)
    round_p_value = round(p_value, 2)
    round_std_err = round(std_err, 2)

    # defining label for regplot
    regplot_label = f'slope: {round_slope}\n'
    regplot_label += f'intercept: {round_intercept}\n'
    regplot_label += f'r_value: {round_r_value}\n'
    regplot_label += f'p_value: {round_p_value}\n'
    regplot_label += f'std_err: {round_std_err}\n'

    # plotting regression plot
    ax = regplot(x=x_col,
                 y=y_col,
                 data=df,
                 ci=None,
                 color='b',
                 line_kws={'label': regplot_label,
                           'linewidth': 1},
                 scatter_kws={'s': 5})

    # adding legend
    ax.legend()

    # showing plot
    plt.show()

    # closing plot
    plt.close()


def get_cell_count_df(df: DataFrame,
                      detection_threshold: float
                      ) -> DataFrame:
    """
    Given a merged detections/annotations data frame,
    returns cell count data frame, of following structure:
    | img_name | model_count | fornma_count |
    | img1.png |     62      |      58      |
    | img2.png |     45      |      51      |
    ...
    :param df: DataFrame. Represents merged detections/annotations data.
    :param detection_threshold: Float. Represents detection threshold to be applied as filter.
    :return: DataFrame. Represents cell count data frame.
    """
    # defining placeholder value for dfs_list
    dfs_list = []

    # grouping df
    df_groups = df.groupby('img_file_name')

    # iterating over df groups
    for img_name, df_group in df_groups:

        # getting current image fornma/model dfs
        fornma_df = df_group[df_group['evaluator'] == 'fornma']
        model_df = df_group[df_group['evaluator'] == 'model']

        # filtering current image model df by detection threshold
        filtered_model_df = model_df[model_df['detection_threshold'] >= detection_threshold]

        # getting current image fornma/model cell counts
        fornma_cell_count = len(fornma_df)
        model_cell_count = len(filtered_model_df)

        # assembling current image dict
        current_group_dict = {'img_name': img_name,
                              'fornma_cell_count': fornma_cell_count,
                              'model_cell_count': model_cell_count}

        # assembling current image df
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
                x='fornma_cell_count',
                y='model_cell_count')

    # setting title/axes names
    plt.title('Cell count comparison')
    plt.xlabel('Fornma Cell Counts (GT)')
    plt.ylabel('Model Cell Counts (detections)')

    # setting axes ticks
    plt.xticks(range(df['fornma_cell_count'].min(), df['fornma_cell_count'].max(), 5))
    plt.yticks(range(df['model_cell_count'].min(), df['model_cell_count'].max(), 5))

    print(df)

    # showing plot
    plt.show()


def compare_model_cell_count_to_gt(detection_file_path: str,
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

    # getting cell count data
    print('getting cell count df...')
    cell_count_df = get_cell_count_df(df=merged_df,
                                      detection_threshold=detection_threshold)

    # plotting cell count data
    print('plotting cell count data...')
    plot_cell_count_data(df=cell_count_df)

    # plotting linear regression
    plot_linear_regression(df=cell_count_df,
                           x_col='fornma_cell_count',
                           y_col='model_cell_count')

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

    # running compare_model_cell_count_to_gt function
    compare_model_cell_count_to_gt(detection_file_path=detection_file,
                                   ground_truth_file_path=ground_truth_file,
                                   detection_threshold=detection_threshold)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
