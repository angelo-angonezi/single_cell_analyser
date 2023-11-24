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
from pandas import read_csv
from seaborn import barplot
from seaborn import lineplot
from pandas import DataFrame
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import enter_to_continue
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


def get_analysis_df(input_path: str) -> DataFrame:
    """
    Given a path to a fornma nucleus csv file,
    returns analysis data frame.
    """
    # reading fornma file
    analysis_df = read_csv(input_path)

    # defining columns to keep
    print(analysis_df)
    exit()

    # returning analysis df
    return analysis_df


def plot_fornma_correlations(input_path: str) -> None:
    """
    Given a path to a fornma output file,
    runs analysis and plots data on screen.
    :param input_path: String. Represents a path to a file.
    :return: None.
    """
    # getting analysis df
    df = get_analysis_df(input_path=input_path)
    print(df)

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
    enter_to_continue()

    # running plot_fornma_correlations function
    plot_fornma_correlations(input_path=input_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
