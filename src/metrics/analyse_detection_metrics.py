# analyse metrics module

print('initializing...')  # noqa

# Code destined to analysing data frame containing
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
from seaborn import scatterplot
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import spacer
from sklearn.metrics import mean_squared_error
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
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


def analyse_metrics(input_path: str,
                    output_folder: str,
                    ) -> None:
    # getting metrics df
    print('getting metrics df...')
    metrics_df = read_csv(input_path)
    print(metrics_df)

    # printing execution message
    print(f'output saved to "{output_folder}"')
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
    enter_to_continue()

    # running plot_metric function
    analyse_metrics(input_path=input_path,
                    output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
