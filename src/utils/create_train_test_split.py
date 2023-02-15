# create train/test split module

print('initializing...')  # noqa

# code destined to generating train/test
# split, based on annotated data.

######################################################################
# importing required libraries

print('importing required libraries...')  # noqa
from os import mkdir
from time import sleep
from os import listdir
from random import seed
from os.path import join
from random import sample
from argparse import ArgumentParser
from src.utils.aux_funcs import spacer
print('all required libraries successfully imported.')  # noqa
sleep(0.8)

######################################################################
# defining global parameters

SEED = 53
SPLIT_RATIO = 0.7
ANNOTATIONS_SUBFOLDERS = ['alpr_format',
                          'dota_format',
                          'lucas_xml_format',
                          'model_output_format',
                          'rolabelimg_format',
                          'tf_records_format']

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'module used to generate train/test split'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # images folder path param
    parser.add_argument('-i', '--images-folder-path',
                        dest='images_folder_path',
                        type=str,
                        help='defines path to folder containing images (.jpg)',
                        required=True)

    # annotations folder path param
    parser.add_argument('-a', '--annotations-folder-path',
                        dest='annotations_folder_path',
                        type=str,
                        help='defines path to folder containing annotations (.xml) [rolabelimg]',
                        required=True)

    # output folder path param
    parser.add_argument('-o', '--output-folder-path',
                        dest='output_folder_path',
                        type=str,
                        help='defines path to folder which will contain "train" and "test" annotated data',
                        required=True)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def create_subfolders_in_folder(folder_path: str,
                                subfolders_list: list
                                ) -> None:
    """
    Given a list of subfolders and a folder path,
    creates subfolders in given folder.
    :param folder_path: String. Represents a path to a folder.
    :param subfolders_list: List. Represents subfolders to be created.
    :return: None.
    """
    # iterating over folders list
    for subfolder in subfolders_list:

        # creating current subfolder path
        subfolder_path = join(folder_path,
                              subfolder)

        # creating current subfolder
        mkdir(subfolder_path)


def create_subfolders_in_output_folder(output_folder_path: str,
                                       annotations_subfolder_list: list
                                       ) -> None:
    """
    Given a path to an output folder, creates
    required subfolders in given output folder.
    :param output_folder_path: String. Represents a path to a folder.
    :param annotations_subfolder_list: List. Represents subfolders to be created.
    :return: None.
    """
    # creating train/test subfolders
    train_subfolder_path = join(output_folder_path,
                                'train')
    test_subfolder_path = join(output_folder_path,
                               'test')
    mkdir(train_subfolder_path)
    mkdir(test_subfolder_path)

    # creating images subfolders
    train_images_subfolder_path = join(train_subfolder_path, 'imgs')
    test_images_subfolder_path = join(test_subfolder_path, 'imgs')
    mkdir(train_images_subfolder_path)
    mkdir(test_images_subfolder_path)

    # creating annotations subfolders
    train_annotations_subfolder_path = join(train_subfolder_path, 'annotations')
    test_annotations_subfolder_path = join(test_subfolder_path, 'annotations')
    mkdir(train_annotations_subfolder_path)
    mkdir(test_annotations_subfolder_path)

    # creating annotations subfolders
    create_subfolders_in_folder(folder_path=train_annotations_subfolder_path,
                                subfolders_list=annotations_subfolder_list)
    create_subfolders_in_folder(folder_path=test_annotations_subfolder_path,
                                subfolders_list=annotations_subfolder_list)


def get_train_test_split_lists(images_names_list: list,
                               train_images_num: int
                               ) -> tuple:
    """
    Given a list of image names (no extension),
    and a split ratio (0.0-1.0) representing
    proportion of train images in data set,
    returns a tuple of two lists, of following
    structure:
    ([train_images_names], [test_images_names])
    :param images_names_list: List. Represents a list of image names.
    :param train_images_num: Integer. Represents number of desired train
    images in train/test dataset split.
    :return: Tuple. Represents train/test split.
    """
    # defining placeholder value for train_test_split_list
    train_test_split_list = []

    # setting seed value
    seed(SEED)

    # getting a sample from images names list
    train_sample = sample(images_names_list,
                          )




def create_train_test_split(images_folder_path: str,
                            annotations_folder_path: str,
                            output_folder_path: str
                            ) -> None:
    """
    Given a path to images and annotations folder,
    creates required subfolders in output folder path,
    and copies images/annotations to subfolders split
    into train/test data sets.
    :param images_folder_path: String. Represents a path to a folder.
    :param annotations_folder_path: String. Represents a path to a folder.
    :param output_folder_path: String. Represents a path to a folder.
    :return: None.
    """
    # creating subfolder in output folder
    print('creating subfolder in output folder...')
    create_subfolders_in_output_folder(output_folder_path=output_folder_path,
                                       annotations_subfolder_list=ANNOTATIONS_SUBFOLDERS)

    # getting images names (no extension) in input folder
    print('getting images in input folder...')
    images = listdir(images_folder_path)
    images_names = [image.replace('.jpg', '')
                    for image
                    in images
                    if image.endswith('.jpg')]
    images_num = len(images_names)

    # getting train/test image numbers based on split ratio
    train_num = int(images_num * SPLIT_RATIO)
    test_num = images_num - train_num

    # printing execution message
    f_string = f'A total of {images_num} images were found in input folder.\n'
    f_string += f'Train imgs: {train_num} ({round(SPLIT_RATIO * 100)}%)\n'
    f_string += f'Test imgs: {test_num} ({round((1 - SPLIT_RATIO) * 100)}%)'
    spacer()
    print(f_string)
    spacer()

    # getting train/test split lists
    print('getting train/test split lists...')
    train_list, test_list = get_train_test_split_lists(images_names_list=images_names,
                                                       train_images_num=SPLIT_RATIO)

    exit()

######################################################################
# defining main function


def main():
    """
    Runs main code.
    """
    # getting data from Argument Parser
    args_dict = get_args_dict()

    # getting images folder path
    images_folder_path = args_dict['images_folder_path']

    # getting annotations folder path
    annotations_folder_path = args_dict['annotations_folder_path']

    # getting output folder path
    output_folder_path = args_dict['output_folder_path']

    # printing execution parameters
    execution_parameters_str = '---Execution Parameters---\n'
    execution_parameters_str += f'images_folder_path: {images_folder_path}\n'
    execution_parameters_str += f'annotations_folder_path: {annotations_folder_path}\n'
    execution_parameters_str += f'output_folder_path: {output_folder_path}'
    spacer()
    print(execution_parameters_str)
    spacer()

    # waiting user input
    i_string = f'Press "Enter" to continue'
    input(i_string)
    spacer()

    # running create_train_test_split function
    create_train_test_split(images_folder_path=images_folder_path,
                            annotations_folder_path=annotations_folder_path,
                            output_folder_path=output_folder_path)


######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
