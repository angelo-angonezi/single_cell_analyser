# ImagesFilter create data splits module

print('initializing...')  # noqa

# Code destined to splitting data for
# ImagesFilter classification network.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from random import shuffle
from argparse import ArgumentParser
from random import seed as set_seed
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import copy_multiple_files
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.1
TEST_SPLIT = 0.3
SEED = 53

# setting seed (so that all executions result in same sample)
set_seed(SEED)

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'ImagesFilter create data splits module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # images folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing source images.')

    # images extension param
    parser.add_argument('-e', '--extension',
                        dest='extension',
                        required=True,
                        help='defines images extension (.png, .jpg, .tif).')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to folder containing train/val/test subfolders.')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def create_data_splits(images_folder: str,
                       extension: str,
                       output_folder: str,
                       data_class: str
                       ) -> None:
    # getting respective data class folders
    input_folder = join(images_folder, data_class)

    # getting images in input folder
    print('getting images in input folder...')
    images = get_specific_files_in_folder(path_to_folder=input_folder,
                                          extension=extension)

    # removing extension (already add later on code) <- required to work with already existent aux_func
    images = [image.replace(extension, '')
              for image
              in images]

    # getting images num
    images_num = len(images)

    # getting split sizes
    train_size = int(TRAIN_SPLIT * images_num)
    val_size = int(VAL_SPLIT * images_num)
    test_size = int(TEST_SPLIT * images_num)

    # getting subfolder paths
    train_folder = join(output_folder, 'train', data_class)
    val_folder = join(output_folder, 'val', data_class)
    test_folder = join(output_folder, 'test', data_class)

    # printing execution message
    f_string = f'found {images_num} images in "{data_class}" input folder.\n'
    f_string += f'{train_size} will be copied to "train" folder.\n'
    f_string += f'{val_size} will be copied to "val" folder.\n'
    f_string += f'{test_size} will be copied to "test" folder.'
    print(f_string)

    # shuffling data (randomizes order)
    print('shuffling images list...')
    shuffle(images)

    # getting splits
    print('creating splits...')
    train_files = images[0: train_size]
    val_files = images[train_size: train_size + val_size]
    test_files = images[train_size + val_size: -1]

    # copying train images
    print('copying train images...')
    copy_multiple_files(src_folder_path=input_folder,
                        dst_folder_path=train_folder,
                        files_list=train_files,
                        file_extension=extension)

    # copying val images
    print('copying val images...')
    copy_multiple_files(src_folder_path=input_folder,
                        dst_folder_path=val_folder,
                        files_list=val_files,
                        file_extension=extension)

    # copying test images
    print('copying test images...')
    copy_multiple_files(src_folder_path=input_folder,
                        dst_folder_path=test_folder,
                        files_list=test_files,
                        file_extension=extension)

    # printing execution message
    print('data splits created!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting images folder param
    images_folder = args_dict['images_folder']

    # getting images extension param
    extension = args_dict['extension']

    # getting output folder param
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # splitting included data
    print('splitting images from class "included"...')
    create_data_splits(images_folder=images_folder,
                       extension=extension,
                       output_folder=output_folder,
                       data_class='included')

    # splitting excluded data
    print('splitting images from class "excluded"...')
    create_data_splits(images_folder=images_folder,
                       extension=extension,
                       output_folder=output_folder,
                       data_class='excluded')

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
