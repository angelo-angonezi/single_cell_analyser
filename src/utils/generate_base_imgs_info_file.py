# generate base imgs info file module

print('initializing...')  # noqa

# Code destined to analysing project's images info file.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from pandas import DataFrame
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
    parser.add_argument('-te', '--test-imgs-folder',
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
                                 images_extension: str,
                                 output_path: str
                                 ) -> None:
    """
    Given a path to train and test images folder,
    and given images extension, generates base
    image info file in output path.
    :param train_imgs_folder: String. Represents a path to a folder.
    :param test_imgs_folder: String. Represents a path to a folder.
    :param images_extension: String. Represents image extension.
    :param output_path: String. Represents output file path.
    """
    # getting files in input folders
    print('getting files in train_imgs_folder...')
    train_imgs = get_specific_files_in_folder(path_to_folder=train_imgs_folder,
                                              extension=images_extension)
    train_imgs_num = len(train_imgs)

    print('getting files in test_imgs_folder...')
    test_imgs = get_specific_files_in_folder(path_to_folder=test_imgs_folder,
                                             extension=images_extension)
    test_imgs_num = len(test_imgs)

    all_imgs = train_imgs.extend(test_imgs)
    imgs_num = len(all_imgs)

    # printing execution message
    f_string = f'A total of {imgs_num} images were found in input folders\n'
    f_string += f'train_imgs_num: {train_imgs_num}'
    f_string += f'test_imgs_num: {test_imgs_num}'
    print(f_string)

    # iterating over images
    for image_index, image_name in enumerate(all_imgs, 1):

        # printing execution message
        progress_ratio = image_index / imgs_num
        progress_percentage = progress_ratio * 100
        progress_percentage_round = round(progress_percentage)
        progress_string = f'analysing image {image_index} of {imgs_num} ({progress_percentage_round}%)'
        flush_or_print(string=progress_string,
                       index=image_index,
                       total=imgs_num)

        # getting current image data
        current_img_split = image_name.split('_')
        current_img_experiment =
        current_img_author =
        current_img_cell_line =
        current_img_treatment =
        current_img_well =
        current_img_frame =

        # creating current image dict
        current_img_dict = {'img_name': image_name}

        # creating current image df
        current_img_df = DataFrame(current_img_dict,
                                   index=[0])

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting train_imgs_folder param
    train_imgs_folder = args_dict['train_imgs_folder']

    # getting test_imgs_folder param
    test_imgs_folder = args_dict['test_imgs_folder']

    # getting images_extension param
    images_extension = args_dict['images_extension']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running analyse_imgs_info_file function
    generate_base_imgs_info_file(train_imgs_folder=train_imgs_folder,
                                 test_imgs_folder=test_imgs_folder,
                                 images_extension=images_extension)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
