# merge channels module

print('initializing...')  # noqa

# Code destined to merging channels
# for forNMA integration.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from numpy import unique
from numpy import ndarray
from seaborn import histplot
from numpy import concatenate
from numpy import add as np_add
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
                       red_normalizer: int,
                       green_normalizer: int,
                       output_path: str
                       ) -> None:
    """
    Given paths for red/green images,
    saves images merge into output path.
    :param red_image_path: String. Represents a path to a file.
    :param green_image_path: String. Represents a path to a file.
    :param red_normalizer: Integer. Represents a normalizer value for red channel.
    :param green_normalizer: Integer. Represents a normalizer value for green channel.
    :param output_path: String. Represents a path to a file.
    :return: None.
    """
    # opening red/green images
    red_image = tiff_imread(red_image_path)
    green_image = tiff_imread(green_image_path)

    # merging images
    merge_image = np_add(red_image, green_image)

    # saving merged image
    tiff_imwrite(file=output_path,
                 data=merge_image)


def merge_multiple_images(red_images_paths: list,
                          green_images_paths: list,
                          red_normalizer: int,
                          green_normalizer: int,
                          output_folder: str
                          ) -> None:
    """
    Given paths for red/green images,
    saves images merge into output path.
    :param red_images_paths: List. Represents paths to files.
    :param green_images_paths: List. Represents paths to files.
    :param red_normalizer: Integer. Represents a normalizer value for red channel.
    :param green_normalizer: Integer. Represents a normalizer value for green channel.
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
                           red_normalizer=red_normalizer,
                           green_normalizer=green_normalizer,
                           output_path=save_path)

    # printing execution message
    f_string = f'all {total_imgs_num} images merged!'
    print(f_string)


def get_single_array(images_paths: list) -> ndarray:
    """
    Given a list paths to images, returns
    single concatenated array of all images.
    :param images_paths: List. Represents paths to files.
    :return: ndarray: Represents all pixels in all images.
    """
    # defining placeholder value for arrays_list
    arrays_list = []

    # iterating over images paths
    for image_path in images_paths:
        # reading image
        open_image = tiff_imread(image_path)

        # flattening image array
        flat_array = open_image.flatten()

        # appending flat array to arrays_list
        arrays_list.append(flat_array)

    # concatenating arrays in arrays_list
    concatenated_array = concatenate(arrays_list)

    # returning concatenated_array
    return concatenated_array


def get_normalization_value(concatenated_array: ndarray) -> int:
    """
    Given a numpy array, returns a normalization
    value, based on array mean/std values.
    :param concatenated_array: ndarray. Represents
    multiple images' arrays joined together.
    :return: Integer. Represents normalization value.
    """
    a = concatenated_array.tolist()
    print(len(a))
    b = sorted(a)
    print(len(b))
    c = b[::1000]
    print(len(c))

    # TODO: rewrite this function to work with arrays all the way through

    # plotting histogram
    histplot(data=c)
    from matplotlib import pyplot as plt
    plt.show()

    exit()

    # calculating normalization value
    normalization_value = 0

    # retuning normalization_value
    return normalization_value


def get_channel_normalization_values(images_paths: list) -> int:
    """
    Given a list paths to images, returns
    normalization values for each channel.
    :param images_paths: List. Represents paths to files.
    :return: Integer. Represents channel normalization value.
    """
    # getting concatenated array from input images
    concatenated_array = get_single_array(images_paths=images_paths)

    # calculating normalization value
    normalization_value = get_normalization_value(concatenated_array=concatenated_array)

    # retuning normalization_value
    return normalization_value


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

        # printing execution message
        e_string = 'red and green images match!'
        print(e_string)

        # getting normalization values
        print('getting normalization values...')
        red_normalizer = get_channel_normalization_values(images_paths=red_images_paths)
        green_normalizer = get_channel_normalization_values(images_paths=green_images_paths)

        # printing execution message
        f_string = 'normalization values acquired.'
        f_string += f'red_normalizer: {red_normalizer}\n'
        f_string += f'green_normalizer: {green_normalizer}'
        print(f_string)

        # merging images
        print('merging images...')
        merge_multiple_images(red_images_paths=red_images_paths,
                              green_images_paths=green_images_paths,
                              red_normalizer=red_normalizer,
                              green_normalizer=green_normalizer,
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
    # enter_to_continue()

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
