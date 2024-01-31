# split channels module

print('initializing...')  # noqa

# Code destined to splitting channels
# for forNMA integration.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import imread
from cv2 import imwrite
from os.path import join
from cv2 import split as cv_split
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
    description = 'split channels module - split red/green data from RGB "as displayed" Incucyte images'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        type=str,
                        help='defines path to folder containing source (RGB .tif) images',
                        required=True)

    # red images folder path param
    parser.add_argument('-r', '--red-folder',
                        dest='red_folder',
                        type=str,
                        help='defines path to folder which will contain red images (8-bit .tif)',
                        required=True)

    # red images folder path param
    parser.add_argument('-g', '--green-folder',
                        dest='green_folder',
                        type=str,
                        help='defines path to folder which will contain green images (8-bit .tif)',
                        required=True)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def split_single_image(input_path: str,
                       red_path: str,
                       green_path: str
                       ) -> None:
    """
    Given a path to an RGB image,
    splits images in red/green arrays,
    saving each array to respective
    given output path.
    """
    # opening input image
    rgb_image = imread(input_path, -1)

    # splitting image
    blue, green, red = cv_split(rgb_image)

    # saving red/green splits
    imwrite(red_path,
            red)
    imwrite(green_path,
            green)


def split_multiple_images(images_list: list,
                          input_folder: str,
                          red_folder: str,
                          green_folder: str
                          ) -> None:
    """
    Given a path to a folder containing
    RGB images, splits images in red/green
    arrays, saving each array to respective
    given output folder.
    """
    # getting images total
    images_num = len(images_list)

    # starting counter for current image index
    current_image_index = 1

    # iterating over images
    for image_name in images_list:

        # printing execution message
        progress_base_string = 'splitting image #INDEX# of #TOTAL#'
        print_progress_message(base_string=progress_base_string,
                               index=current_image_index,
                               total=images_num)

        # getting current image input path
        input_path = join(input_folder,
                          image_name)

        # getting current image output paths
        red_path = join(red_folder,
                        image_name)
        green_path = join(green_folder,
                          image_name)

        # running split_single_image function
        split_single_image(input_path=input_path,
                           red_path=red_path,
                           green_path=green_path)

    # printing execution message
    f_string = f'all {images_num} images split!'
    print(f_string)


def split_channels(input_folder: str,
                   red_folder: str,
                   green_folder: str
                   ) -> None:
    """
    Given paths for red/green images folders,
    splits each image based on both channels,
    and saves them to given output folder.
    """
    # getting images in input folder
    print('getting images in input folder...')
    input_images = get_specific_files_in_folder(path_to_folder=input_folder,
                                                extension='.tif')

    # splitting images
    print('splitting images...')
    split_multiple_images(images_list=input_images,
                          input_folder=input_folder,
                          red_folder=red_folder,
                          green_folder=green_folder)

######################################################################
# defining main function


def main():
    """
    Gets arguments from cli and runs main code.
    """
    # getting data from Argument Parser
    args_dict = get_args_dict()

    # getting input images folder param
    input_folder = args_dict['input_folder']

    # getting red images folder param
    red_folder = args_dict['red_folder']

    # getting green images folder param
    green_folder = args_dict['green_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running split_channels function
    split_channels(input_folder=input_folder,
                   red_folder=red_folder,
                   green_folder=green_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
