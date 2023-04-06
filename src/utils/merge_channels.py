# merge channels module

print('initializing...')  # noqa

# Code destined to merging channels
# for forNMA integration.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from argparse import ArgumentParser
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'merge channels module - join red/green data into single image'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # red images folder path param
    parser.add_argument('-r', '--red-folder-path',
                        dest='red_folder_path',
                        type=str,
                        help='defines path to folder containing red images (.tif)',
                        required=True)

    # red images folder path param
    parser.add_argument('-g', '--green-folder-path',
                        dest='green_folder_path',
                        type=str,
                        help='defines path to folder containing green images (.tif)',
                        required=True)

    # output folder path param
    parser.add_argument('-o', '--output-folder-path',
                        dest='output_folder_path',
                        type=str,
                        help='defines path to folder which will contain merged images (.tif)',
                        required=True)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def merge_single_image(red_image_path: str,
                       green_image_path: str,
                       output_path: str
                       ) -> None:
    """
    Given paths for red/green images,
    saves images merge into output path.
    :param red_image_path: String. Represents a path to a file.
    :param green_image_path: String. Represents a path to a file.
    :param output_path: String. Represents a path to a file.
    :return: None.
    """
    pass


def merge_multiple_images(red_images_folder: str,
                          green_images_folder: str,
                          output_folder: str
                          ) -> None:
    """
    Given paths for red/green images folders,
    merges each image based on both channels,
    and saves them to given output folder.
    :param red_images_folder: String. Represents a path to a folder.
    :param green_images_folder: String. Represents a path to a folder.
    :param output_folder: String. Represents a path to a folder.
    :return: None.
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

    # getting red images folder param
    red_folder_path = args_dict['red_folder_path']

    # getting green images folder param
    green_folder_path = args_dict['green_folder_path']

    # getting output folder param
    output_folder_path = args_dict['output_folder_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running asdasdasdasd function


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module