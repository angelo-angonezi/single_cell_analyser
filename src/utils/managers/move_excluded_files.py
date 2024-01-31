# move hard files module

print('initializing...')  # noqa

# Code destined to moving files according
# to an inclusion/exclusion table.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from random import seed as set_seed
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_image_confluence
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'move hard files module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # table param
    parser.add_argument('-t', '--table-path',
                        dest='table_path',
                        required=True,
                        help='defines path to table (.csv) containing included/excluded info')

    # input folder param
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help='defines path to folder containing input images')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to folder containing "included" and "excluded" subfolder')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def move_excluded_files(table_path: str,
                        input_folder: str,
                        output_folder: str
                        ) -> None:
    """
    Given a path to a table containing included/excluded
    info, copies files from input folder to respective
    subfolders in output folder.
    """
    # reading table
    files_df = read_csv(table_path)

    print(files_df)
    exit()


######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input table param
    table_path = args_dict['table_path']

    # getting input folder param
    input_folder = args_dict['input_folder']

    # getting output folder param
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running move_excluded_files function
    move_excluded_files(table_path=table_path,
                        input_folder=input_folder,
                        output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
