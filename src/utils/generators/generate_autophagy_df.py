# generate autophagy df module

print('initializing...')  # noqa

# Code destined to generating autophagy
# data frame, based on segmentation masks.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import get_crop_pixels
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
    description = 'generate autophagy df module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # red folder param
    parser.add_argument('-r', '--red-folder',
                        dest='red_folder',
                        required=True,
                        help='defines red input folder (folder containing crops in fluorescence channel)')

    # green folder param
    parser.add_argument('-g', '--green-folder',
                        dest='green_folder',
                        required=True,
                        help='defines green input folder (folder containing crops in fluorescence channel)')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # crops file param
    crops_help = 'defines path to crops file (containing crops info)'
    parser.add_argument('-c', '--crops-file',
                        dest='crops_file',
                        required=True,
                        help=crops_help)

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output file (.csv)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def generate_autophagy_df(red_folder: str,
                          green_folder: str,
                          images_extension: str,
                          crops_file: str,
                          output_path: str,
                          ) -> None:
    """
    TODO: update!
    Given a path to a folder containing crops,
    and a path to a file containing crops info,
    generates crops pixels data frame, and saves
    it to given output path.
    """

    # printing execution message
    print(f'output saved to {output_path}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting red folder
    red_folder = args_dict['red_folder']

    # getting green folder
    green_folder = args_dict['green_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting crops file
    crops_file = args_dict['crops_file']

    # getting output path
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_autophagy_df function
    generate_autophagy_df(red_folder=red_folder,
                          green_folder=green_folder,
                          images_extension=images_extension,
                          crops_file=crops_file,
                          output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
