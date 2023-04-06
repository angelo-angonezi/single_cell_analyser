# compare model nma to ground-truth module

print('initializing...')  # noqa

# Code destined to comparing NMA results
# between model detections and gt annotations.

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
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_merged_detection_annotation_df
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "compare model nma to gt module"

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


def add_cell_area_col(df: DataFrame) -> None:
    """
    Given a merged detections/annotations data frame,
    adds 'cell_area' column, calculated by
    multiplying width/height cols.
    :param df: DataFrame. Represents merged detections/annotations data.
    :return: None.
    """
    # adding cell area column to df
    df['cell_area'] = df['width'] * df['height']


def get_axis_ratio(width: float,
                   height: float
                   ) -> float:
    """
    Given width and height values, checks which one
    is larger, and returns ratio between longer
    and shorter axis.
    :param width: Float. Represents OBB width.
    :param height: Float. Represents OBB height.
    :return: Float. Represents axis ratio.
    """
    # defining long and short axis based on width/height values
    long_axis = width if width > height else height
    short_axis = width if width < height else height

    # calculating axis_ratio
    axis_ratio = long_axis / short_axis

    # returning axis_ratio
    return axis_ratio


def add_axis_ratio_col(df: DataFrame) -> None:
    """
    Given a merged detections/annotations data frame,
    adds 'cell_area' column, calculated by dividing
    width/height cols (order varies depending on
    which is larger).
    :param df: DataFrame. Represents merged detections/annotations data.
    :return: None.
    """
    # adding axis ratio placeholder column to df
    df['axis_ratio'] = None

    # getting df rows
    df_rows = df.iterrows()

    # iterating over df rows
    for row_index, row_data in df_rows:

        # getting current row width/height data
        current_width = row_data['width']
        current_height = row_data['height']

        # getting axis ratio
        current_axis_ratio = get_axis_ratio(width=current_width,
                                            height=current_height)

        # updating current line axis ratio value
        df.at[row_index, 'axis_ratio'] = current_axis_ratio


def add_treatment_col(df: DataFrame) -> None:
    """
    Given a merged detections/annotations data frame,
    adds 'treatment' column, obtained by file name.
    :param df: DataFrame. Represents merged detections/annotations data.
    :return: None.
    """
    # adding treatment placeholder column to df
    df['treatment'] = None

    # getting df rows
    df_rows = df.iterrows()

    # iterating over df rows
    for row_index, row_data in df_rows:

        # getting current row treatment data
        img_file_name = row_data['img_file_name']
        img_file_name_split = img_file_name.split('_')
        treatment_col = img_file_name_split[1]
        treatment_str = treatment_col[0]

        # defining current treatment
        current_treatment = 'Control' if treatment_str == 'B' else 'ATF6'

        # updating current line axis ratio value
        df.at[row_index, 'treatment'] = current_treatment


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
    add_treatment_col(df=df)

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


def plot_annotations_histograms(df: DataFrame) -> None:
    """
    Given a nma data frame, plots cell_area and axis_ratio
    histograms, filtering df by control group.
    :param df: DataFrame. Represents NMA data.
    :return: None.
    """
    # filtering df for annotations data only
    annotator_df = df[df['evaluator'] == 'fornma']

    # defining x_cols
    x_cols = ['cell_area', 'axis_ratio']

    # iterating over x_cols
    for x_col in x_cols:

        # creating histogram
        histplot(data=annotator_df,
                 x=x_col,
                 hue='treatment',
                 kde=True)
    
        # showing plot
        plt.show()

        # closing plot
        plt.close()


def plot_nma_data(df: DataFrame) -> None:
    """
    Given a nma data frame, plots nma (cell_area X axis_ratio),
    coloring data by evaluator (model detections and fornma
    annotations).
    :param df: DataFrame. Represents NMA data.
    :return: None.
    """
    # creating grid for plot
    plot_grid = FacetGrid(data=df,
                          col='treatment',
                          hue='evaluator')

    # mapping scatter plot
    plot_grid.map(scatterplot,
                  'axis_ratio',
                  'cell_area')

    # adding legend
    plot_grid.add_legend()

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

    # getting nma data
    print('getting nma df...')
    nma_df = get_nma_df(df=merged_df)

    # adding manual values
    manual_df = read_csv('/home/angelo/Desktop/fer_nma_data.csv',
                         decimal=',')
    print(manual_df)
    print(nma_df)
    dfs_list = [manual_df, nma_df]
    final_df = concat(dfs_list)

    # plotting nma data
    print('plotting nma histograms...')
    plot_annotations_histograms(df=nma_df)

    # plotting nma data
    print('plotting nma...')
    plot_nma_data(df=nma_df)

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
