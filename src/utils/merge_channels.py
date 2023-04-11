# merge channels module

print('initializing...')  # noqa

# Code destined to merging channels
# for forNMA integration.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from numpy import add as np_add
from cv2 import imread
from cv2 import IMREAD_UNCHANGED
from argparse import ArgumentParser
from os.path import split as os_split
from tifffile import imread as tiff_imread
from tifffile import imwrite as tiff_imwrite
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


def red_matches_green(red_images: list,
                      green_images: list
                      ) -> bool:
    """
    Given red/green images lists, returns True
    if all images contained in folders match,
    and False otherwise.
    """
    # checking whether image names match
    match_bool = (red_images == green_images)

    # returning boolean value
    return match_bool


def merge_single_image(red_image_path: str,
                       green_image_path: str,
                       output_path: str,
                       red_multiplier: int
                       ) -> None:
    """
    Given paths for red/green images,
    saves images merge into output path.
    :param red_image_path: String. Represents a path to a file.
    :param green_image_path: String. Represents a path to a file.
    :param output_path: String. Represents a path to a file.
    :param red_multiplier: Integer. Represents red channel weight.
    :return: None.
    """
    # opening red/green images
    red_image = tiff_imread(red_image_path)

    green_image = tiff_imread(green_image_path)

    # adding weights to red channel
    red_image = red_image * red_multiplier

    # merging images
    merge_image = np_add(red_image, green_image)

    # saving merged image
    tiff_imwrite(file=output_path,
                 data=merge_image)


def merge_multiple_images(red_images_paths: list,
                          green_images_paths: list,
                          output_folder: str
                          ) -> None:
    """
    Given paths for red/green images,
    saves images merge into output path.
    :param red_images_paths: List. Represents paths to files.
    :param green_images_paths: List. Represents paths to files.
    :param output_folder: String. Represents a path to a folder.
    :return: None.
    """
    # getting total imgs num
    total_imgs_num = len(red_images_paths)

    # creating images zip
    red_green_zip = zip(red_images_paths, green_images_paths)

    # iterating over zip items
    for img_index, (red_image_path, green_image_path) in enumerate(red_green_zip, 1):

        # printing execution message
        progress_base_string = 'merging image #INDEX# of #TOTAL#'
        print_progress_message(base_string=progress_base_string,
                               index=img_index,
                               total=total_imgs_num)

        # getting save path
        name_split = os_split(red_image_path)
        save_name = name_split[-1]
        save_name = str(save_name)
        save_path = join(output_folder,
                         save_name)

        # running merge_single_image function
        merge_single_image(red_image_path=red_image_path,
                           green_image_path=green_image_path,
                           output_path=save_path,
                           red_multiplier=3)

    # printing execution message
    f_string = f'all {total_imgs_num} images merged!'
    print(f_string)


def merge_channels(red_images_folder: str,
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
    # getting images in red/green folders
    red_images = get_specific_files_in_folder(path_to_folder=red_images_folder,
                                              extension='.tif')
    green_images = get_specific_files_in_folder(path_to_folder=green_images_folder,
                                                extension='.tif')

    # getting images num
    red_images_num = len(red_images)
    green_images_num = len(green_images)

    # getting images paths
    red_images_paths = [join(red_images_folder, img_name)
                        for img_name
                        in red_images]
    green_images_paths = [join(green_images_folder, img_name)
                          for img_name
                          in green_images]

    # printing execution message
    f_string = f'{red_images_num} red images found in input folder.\n'
    f_string += f'{green_images_num} green images found in input folder.'
    print(f_string)

    # checking whether image names match
    if red_matches_green(red_images=red_images,
                         green_images=green_images):

        # printing error message
        e_string = 'red and green images match!'
        print(e_string)

        # running merge_multiple_images functions
        print('merging images...')
        merge_multiple_images(red_images_paths=red_images_paths,
                              green_images_paths=green_images_paths,
                              output_folder=output_folder)

    # if images do not match
    else:

        # printing error message
        e_string = "red and green images don't match!\n"
        e_string += 'Please, check input folders and try again.'
        print(e_string)

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

    # running merge_channels function
    merge_channels(red_images_folder=red_folder_path,
                   green_images_folder=green_folder_path,
                   output_folder=output_folder_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
