# plot fornma correlations module

print('initializing...')  # noqa

# Code destined to analysing correlations
# between fornma_area VS obb_area and
# fornma_NII VS obb_axis_ratio.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from pandas import Series
from os.path import exists
from pandas import read_csv
from pandas import DataFrame
from seaborn import scatterplot
from scipy.stats import pearsonr
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import get_mask_area
from src.utils.aux_funcs import get_axis_ratio
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'plot fornma correlations module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input path param
    parser.add_argument('-i', '--input-path',
                        dest='input_path',
                        required=True,
                        help='defines path to input file (fornma nucleus output .csv)')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to folder which will contain output files.')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_fornma_area(row_data: Series) -> int:
    """
    Given a fornma df row data,
    returns fornma area.
    """
    # getting fornma area
    fornma_area = row_data['Area']

    # converting area to int
    fornma_area = int(fornma_area)

    # returning fornma area
    return fornma_area


def get_ebb_area(row_data: Series) -> int:
    """
    Given a fornma df row data,
    returns ebb area.
    """
    # getting ebb area
    ebb_area = get_mask_area(row_data=row_data,
                             style='ellipse')

    # returning ebb area
    return ebb_area


def get_fornma_nii(row_data: Series) -> float:
    """
    Given a fornma df row data,
    returns fornma NII.
    """
    # getting fornma NII
    fornma_nii = row_data['NII']

    # returning fornma NII
    return fornma_nii


def get_ebb_axis_ratio(row_data: Series) -> float:
    """
    Given a fornma df row data,
    returns ebb axis ratio (NII).
    """
    # getting row values
    width = row_data['width']
    height = row_data['height']

    # getting axis ratio
    axis_ratio = get_axis_ratio(width=width,
                                height=height)

    # returning axis ratio
    return axis_ratio


def get_fornma_df(input_path: str) -> DataFrame:
    """
    Given a path to a fornma nucleus csv file,
    returns fornma data frame with specific columns.
    """
    # reading fornma file
    fornma_df = read_csv(input_path)

    # defining cols to keep
    cols_to_keep = ['Area',
                    'NII',
                    'FitEllipse_X',
                    'FitEllipse_Y',
                    'FitEllipse_a',
                    'FitEllipse_b',
                    'FitEllipse_angle']

    # dropping unrequired columns
    filtered_df = fornma_df[cols_to_keep]

    # defining new cols names
    new_cols = ['Area',
                'NII',
                'cx',
                'cy',
                'width',
                'height',
                'angle']

    # renaming cols
    filtered_df.columns = new_cols

    # returning filtered df
    return filtered_df


def create_analysis_df(df: DataFrame) -> DataFrame:
    """
    Given a fornma data frame,
    returns analysis data frame.
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # getting df rows
    df_rows = df.iterrows()

    # getting progress total
    progress_total = len(df)

    # defining starter for progress_index
    progress_index = 1

    # iterating over df rows
    for row_index, row_data in df_rows:

        # printing execution message
        base_string = f'getting data for row #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=progress_index,
                               total=progress_total)

        # getting current row values
        current_fornma_area = get_fornma_area(row_data=row_data)
        current_ebb_area = get_ebb_area(row_data=row_data)
        current_fornma_nii = get_fornma_nii(row_data=row_data)
        current_ebb_axis_ratio = get_ebb_axis_ratio(row_data=row_data)

        # assembling current row dict
        current_dict = {'fornma_area': current_fornma_area,
                        'ebb_area': current_ebb_area,
                        'fornma_nii': current_fornma_nii,
                        'ebb_axis_ratio': current_ebb_axis_ratio}

        # assembling current row df
        current_df = DataFrame(current_dict,
                               index=[0])

        # appending current df to dfs list
        dfs_list.append(current_df)

        # updating progress index
        progress_index += 1

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning analysis df
    return final_df


def get_analysis_df(fornma_df: DataFrame,
                    output_folder: str
                    ) -> DataFrame:
    """
    Given a fornma data frame, and a path
    to an output folder, checks whether analysis
    df already exists in given output folder,
    returning it if already existent, or creating
    it based on fornma df and saving it to output
    folder, otherwise.
    """
    # defining save path
    save_name = 'analysis_df.csv'
    save_path = join(output_folder,
                     save_name)

    # defining placeholder value for analysis df
    analysis_df = None

    # attempting to read analysis df
    if exists(save_path):

        # reading analysis df
        print('reading already existing analysis df...')
        analysis_df = read_csv(save_path)
    else:

        # creating analysis df
        print('creating analysis df...')
        analysis_df = create_analysis_df(df=fornma_df)

        # saving analysis df
        print('saving analysis df...')
        analysis_df.to_csv(save_path,
                           index=False)

    # returning analysis df
    return analysis_df


def plot_correlation(df: DataFrame,
                     fornma_col: str,
                     ebb_col: str
                     ) -> None:
    """
    Given an analysis df, plots
    given correlations (with R value).
    """
    # getting fornma/ebb values
    fornma_values = df[fornma_col]
    ebb_values = df[ebb_col]

    # getting area (Pearson) correlation value
    pearson_result = pearsonr(x=fornma_values,
                              y=ebb_values)

    # getting correlation and p values
    corr_value, p_value = pearson_result

    # plotting results
    scatterplot(data=df,
                x=fornma_col,
                y=ebb_col)

    # setting plot title
    title = f'forNMAxEBB correlation plot (Pearson R: {corr_value} | p-value: {p_value})'
    plt.title(title)

    # showing plot
    plt.show()


def plot_area_correlation(df: DataFrame) -> None:
    """
    Given an analysis df, plots
    area correlations (with R value).
    """
    # plotting correlation
    plot_correlation(df=df,
                     fornma_col='fornma_area',
                     ebb_col='ebb_area')


def plot_nii_correlation(df: DataFrame) -> None:
    """
    Given an analysis df, plots
    NII correlations (with R value).
    """
    # plotting correlation
    plot_correlation(df=df,
                     fornma_col='fornma_nii',
                     ebb_col='ebb_axis_ratio')


def plot_fornma_correlations(input_path: str,
                             output_folder: str
                             ) -> None:
    """
    Given a path to a fornma output file,
    runs analysis and plots data on screen,
    saving results in given output folder.
    """
    # getting fornma df
    print('getting fornma df...')
    fornma_df = get_fornma_df(input_path=input_path)

    # getting analysis df
    print('getting analysis df...')
    analysis_df = get_analysis_df(fornma_df=fornma_df,
                                  output_folder=output_folder)

    # plotting area correlation
    print('plotting area correlation...')
    plot_area_correlation(df=analysis_df)

    # plotting NII/axis_ratio correlation
    print('plotting NII/axis_ratio correlation...')
    plot_nii_correlation(df=analysis_df)

    # printing execution message
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input path
    input_path = args_dict['input_path']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running plot_fornma_correlations function
    plot_fornma_correlations(input_path=input_path,
                             output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
