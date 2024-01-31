# merge channels module

print('initializing...')  # noqa

# Code destined to merging channels
# for forNMA integration.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import imread
from cv2 import imwrite
from os.path import join
from numpy import add as np_add
from argparse import ArgumentParser
from numpy import uint8 as np_uint8
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
    description = 'merge channels module - merge red/green data from RGB "as displayed" Incucyte images'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # red images folder path param
    parser.add_argument('-r', '--red-folder',
                        dest='red_folder',
                        type=str,
                        help='defines path to folder containing red images (8-bit .tif)',
                        required=True)

    # red images folder path param
    parser.add_argument('-g', '--green-folder',
                        dest='green_folder',
                        type=str,
                        help='defines path to folder containing green images (8-bit .tif)',
                        required=True)

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        type=str,
                        help='defines path to folder which will contain merged (8-bit .tif) images',
                        required=True)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def merge_single_image(red_path: str,
                       green_path: str,
                       output_path: str
                       ) -> None:
    """
    Given a path to red/green images,
    merges them into single array,
    saving it to given output path.
    """
    # reading red/green images
    red = imread(red_path, -1)
    green = imread(green_path, -1)

    # getting image halves (important to join them later and max still be 255)
    red_half = red / 2
    green_half = green / 2

    # merging images
    merged = np_add(red_half, green_half)

    # converting int type
    merged = merged.astype(np_uint8)

    # saving merged image
    imwrite(output_path,
            merged)


def merge_multiple_images(images_list: list,
                          red_folder: str,
                          green_folder: str,
                          output_folder: str
                          ) -> None:
    """
    Given a path to folders containing
    red/green images, merges images into
    single array, saving output image in
    given output folder.
    """
    # getting images total
    images_num = len(images_list)

    # starting counter for current image index
    current_image_index = 1

    # iterating over images
    for image_name in images_list:

        if image_name != 'u87_fucci_tmz_C3_9_04d00h00m.tif':
            continue

        # printing execution message
        progress_base_string = 'merging image #INDEX# of #TOTAL#'
        print_progress_message(base_string=progress_base_string,
                               index=current_image_index,
                               total=images_num)

        # getting current image input paths
        red_path = join(red_folder,
                        image_name)
        green_path = join(green_folder,
                          image_name)

        # getting current image save path
        save_path = join(output_folder,
                         image_name)

        # running merge_single_image function
        merge_single_image(red_path=red_path,
                           green_path=green_path,
                           output_path=save_path)

        # updating current image index
        current_image_index += 1

    # printing execution message
    f_string = f'all {images_num} images merged!'
    print(f_string)


def merge_channels(output_folder: str,
                   red_folder: str,
                   green_folder: str
                   ) -> None:
    """
    Given paths for red/green images folders,
    merges each image based on both channels,
    and saves them to given output folder.
    """
    # getting images in input folder
    print('getting images in input folder...')
    input_images = get_specific_files_in_folder(path_to_folder=red_folder,
                                                extension='.tif')

    # merging images
    print('merging images...')
    merge_multiple_images(images_list=input_images,
                          red_folder=red_folder,
                          green_folder=green_folder,
                          output_folder=output_folder)

######################################################################
# defining main function


def main():
    """
    Gets arguments from cli and runs main code.
    """
    # getting data from Argument Parser
    args_dict = get_args_dict()

    # getting red images folder param
    red_folder = args_dict['red_folder']

    # getting green images folder param
    green_folder = args_dict['green_folder']

    # getting output images folder param
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running merge_channels function
    merge_channels(output_folder=output_folder,
                   red_folder=red_folder,
                   green_folder=green_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
