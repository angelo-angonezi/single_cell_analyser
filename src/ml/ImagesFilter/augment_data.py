# ImagesFilter augment data module

print('initializing...')  # noqa

# Code destined to augmenting data for
# ImagesFilter classification network.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import imread
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
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
    description = 'ImagesFilter data augmentation module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # images folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing images to be augmented.')

    # images extension param
    parser.add_argument('-e', '--extension',
                        dest='extension',
                        required=True,
                        help='defines images extension (.png, .jpg, .tif).')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to folder which will contain augmented images.')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def augment_data(images_folder: str,
                 extension: str,
                 output_folder: str
                 ) -> None:
    # getting images in input folder
    images = get_specific_files_in_folder(path_to_folder=images_folder,
                                          extension=extension)
    print(images)

    # printing execution message
    print('analysis complete!')
    print(f'results saved to "{output_folder}".')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting images folder param
    images_folder = str(args_dict['images_folder'])

    # getting images extension param
    extension = str(args_dict['extension'])

    # getting output folder param
    output_folder = str(args_dict['output_folder'])

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running image_filter_test function
    augment_data(images_folder=images_folder,
                 extension=extension,
                 output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
