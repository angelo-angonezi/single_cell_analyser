# generate base imgs info file module

print('initializing...')  # noqa

# Code destined to analysing project's images info file.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from argparse import ArgumentParser
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import flush_or_print
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
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
    description = "analyse imgs info file module"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # train_imgs_folder param
    train_imgs_folder_help = 'defines path to folder containing train images'
    parser.add_argument('-tr', '--train-imgs-folder',
                        dest='train_imgs_folder',
                        required=True,
                        help=train_imgs_folder_help)

    # test_imgs_folder param
    test_imgs_folder_help = 'defines path to folder containing test images'
    parser.add_argument('-te', '--train-imgs-folder',
                        dest='test_imgs_folder',
                        required=True,
                        help=test_imgs_folder_help)

    # images_extension param
    images_extension_help = 'defines extension (.tif, .png, .jpg) of images in input folders'
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help=images_extension_help)

    # output_path param
    output_path_help = 'defines path to output csv'
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help=output_path_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def generate_base_imgs_info_file(train_imgs_folder: str,
                                 test_imgs_folder: str,
                                 ) -> None:
    pass

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input file
    input_file = args_dict['input_file']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running analyse_imgs_info_file function
    analyse_imgs_info_file(input_file=input_file)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
