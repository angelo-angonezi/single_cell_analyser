# check incucyte export module

print('initializing...')  # noqa

# Code destined to checking incucyte
# exported images.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os import listdir
from os.path import join
from random import sample
from argparse import ArgumentParser
from random import seed as set_seed
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import create_folder
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import copy_multiple_files
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import create_subfolders_in_folder
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'check incucyte export (if phase/red images match)'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # phase images folder path param
    parser.add_argument('-p', '--phase-folder-path',
                        dest='phase_folder_path',
                        type=str,
                        help='defines path to folder containing phase images (.jpg)',
                        required=True)

    # red images folder path param
    parser.add_argument('-r', '--red-folder-path',
                        dest='red_folder_path',
                        type=str,
                        help='defines path to folder containing red images (.tif)',
                        required=True)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def phase_matches_red(phase_images_folder: str,
                      red_images_folder: str
                      ) -> bool:
    """
    Given a path to phase/red folders, returns True
    if all images contained in folders match, and
    False otherwise.
    """
    # getting
    pass


def check_incucyte_export(phase_images_folder: str,
                          red_images_folder: str
                          ) -> None:
    """
    Given a path to phase/red folders,
    """
    pass

######################################################################
# defining main function


def main():
    """
    Gets arguments from cli and runs main code.
    """
    # getting data from Argument Parser
    args_dict = get_args_dict()

    # getting images folder path
    images_folder_path = args_dict['images_folder_path']

    # getting annotations folder path
    annotations_folder_path = args_dict['annotations_folder_path']

    # getting output folder path
    output_folder_path = args_dict['output_folder_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running create_train_test_split function
    create_train_test_split(images_folder_path=images_folder_path,
                            annotations_folder_path=annotations_folder_path,
                            output_folder_path=output_folder_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
