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
from pandas import read_csv
from seaborn import barplot
from seaborn import lineplot
from pandas import DataFrame
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


def get_analysis_df(df: DataFrame) -> DataFrame:
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


def plot_area_correlation(df: DataFrame) -> None:
    pass


def plot_nii_correlation(df: DataFrame) -> None:
    pass


def plot_fornma_correlations(input_path: str) -> None:
    """
    Given a path to a fornma output file,
    runs analysis and plots data on screen.
    :param input_path: String. Represents a path to a file.
    :return: None.
    """
    # getting fornma df
    fornma_df = get_fornma_df(input_path=input_path)

    # getting analysis df
    analysis_df = get_analysis_df(df=fornma_df)

    # plotting area correlation
    plot_area_correlation(df=analysis_df)

    # plotting nii correlation
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
    input_path = str(args_dict['input_path'])

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running plot_fornma_correlations function
    plot_fornma_correlations(input_path=input_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
