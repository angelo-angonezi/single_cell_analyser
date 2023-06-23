# create train/test split module

print('initializing...')  # noqa

# Code destined to generating train/test
# split, based on annotated data and imgs
# info file.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import create_folder
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import copy_multiple_files
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import create_subfolders_in_folder
print('all required libraries successfully imported.')  # noqa

######################################################################
# defining global parameters

ANNOTATIONS_SUBFOLDERS = ['alpr_format',
                          'dota_format',
                          'pascal_format',
                          'rolabelimg_format',
                          'tfrecords_format']

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

    # images info file param
    parser.add_argument('-f', '--images-info-file-path',
                        dest='images_info_file_path',
                        type=str,
                        help='defines path to file containing images info - which dataset they belong to (.csv)',
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
    create_folder(train_subfolder_path)
    create_folder(test_subfolder_path)

    # creating images subfolders
    train_images_subfolder_path = join(train_subfolder_path, 'imgs')
    test_images_subfolder_path = join(test_subfolder_path, 'imgs')
    create_folder(train_images_subfolder_path)
    create_folder(test_images_subfolder_path)

    # creating annotations subfolders
    train_annotations_subfolder_path = join(train_subfolder_path, 'annotations')
    test_annotations_subfolder_path = join(test_subfolder_path, 'annotations')
    create_folder(train_annotations_subfolder_path)
    create_folder(test_annotations_subfolder_path)

    # creating annotations subfolders
    create_subfolders_in_folder(folder_path=train_annotations_subfolder_path,
                                subfolders_list=annotations_subfolder_list)
    create_subfolders_in_folder(folder_path=test_annotations_subfolder_path,
                                subfolders_list=annotations_subfolder_list)


def get_train_test_split_lists(images_info_df: DataFrame) -> tuple:
    """
    Given a list of image names (no extension),
    and a split ratio (0.0-1.0) representing
    proportion of train images in data set,
    returns a tuple of two lists, of following
    structure:
    ([train_images_names], [test_images_names])
    :param images_info_df: List. Represents a list of image names.
    :return: Tuple. Represents train/test split.
    """
    # getting number of images found in info df
    images_num = len(images_info_df)

    # getting train/test dfs
    train_df = images_info_df[images_info_df['dataset'] == 'train']
    test_df = images_info_df[images_info_df['dataset'] == 'test']

    # getting train/test image nums
    train_num = len(train_df)
    test_num = len(test_df)

    # getting train/test image ratios
    train_ratio = train_num / images_num
    test_ratio = test_num / images_num

    # getting train/test image percentages
    train_percentage = round(train_ratio * 100)
    test_percentage = round(test_ratio * 100)

    # printing execution message
    f_string = f'A total of {images_num} images were found in input folder.\n'
    f_string += f'Train imgs: {train_num} ({train_percentage}%)\n'
    f_string += f'Test imgs: {test_num} ({test_percentage}%)'
    spacer()
    print(f_string)
    spacer()

    # getting train images' names
    train_images = train_df['img_name']
    test_images = test_df['img_name']

    # adding samples to final tuple
    final_tuple = (train_images, test_images)
    print('train/test set created!')

    # returning final tuple
    return final_tuple


def copy_images_and_annotations(images_folder_path: str,
                                annotations_folder_path: str,
                                output_folder_path: str,
                                train_list: list,
                                test_list: list
                                ) -> None:
    """
    Given paths to annotations and images, copies
    files into respective output folder, based on
    given train and test lists.
    :param images_folder_path: String. Represents a path to a folder.
    :param annotations_folder_path: String. Represents a path to a folder.
    :param output_folder_path: String. Represents a path to a folder.
    :param train_list: List. Represents train set image names.
    :param test_list: Represents test set image names.
    :return: None.
    """
    # defining dst folder paths
    train_images_dst_folder = join(output_folder_path,
                                   'train',
                                   'imgs')
    test_images_dst_folder = join(output_folder_path,
                                  'test',
                                  'imgs')
    train_annotations_dst_folder = join(output_folder_path,
                                        'train',
                                        'annotations',
                                        'rolabelimg_format')
    test_annotations_dst_folder = join(output_folder_path,
                                       'test',
                                       'annotations',
                                       'rolabelimg_format')

    # iterating over train images
    spacer()
    print('copying train images to train folder...')
    copy_multiple_files(src_folder_path=images_folder_path,
                        dst_folder_path=train_images_dst_folder,
                        files_list=train_list,
                        file_extension='.jpg')

    spacer()
    print('copying train annotations to train folder...')
    copy_multiple_files(src_folder_path=annotations_folder_path,
                        dst_folder_path=train_annotations_dst_folder,
                        files_list=train_list,
                        file_extension='.xml')

    # iterating over test images
    spacer()
    print('copying test images to test folder...')
    copy_multiple_files(src_folder_path=images_folder_path,
                        dst_folder_path=test_images_dst_folder,
                        files_list=test_list,
                        file_extension='.jpg')

    spacer()
    print('copying test annotations to test folder...')
    copy_multiple_files(src_folder_path=annotations_folder_path,
                        dst_folder_path=test_annotations_dst_folder,
                        files_list=test_list,
                        file_extension='.xml')

    # printing execution message
    p_string = 'all images/annotations files successfully copied to split folder!'
    spacer()
    print(p_string)


def create_train_test_split(images_folder_path: str,
                            annotations_folder_path: str,
                            images_info_file_path: str,
                            output_folder_path: str
                            ) -> None:
    """
    Given a path to images and annotations folder,
    creates required subfolders in output folder path,
    and copies images/annotations to subfolders split
    into train/test data sets.
    :param images_folder_path: String. Represents a path to a folder.
    :param annotations_folder_path: String. Represents a path to a folder.
    :param images_info_file_path: String. Represents a path to a file.
    :param output_folder_path: String. Represents a path to a folder.
    :return: None.
    """
    # creating subfolder in output folder
    print('creating subfolder in output folder...')
    create_subfolders_in_output_folder(output_folder_path=output_folder_path,
                                       annotations_subfolder_list=ANNOTATIONS_SUBFOLDERS)

    # getting images info df
    print('getting images info df...')
    images_info_df = read_csv(images_info_file_path)

    # getting train/test split lists
    print('getting train/test split lists...')
    train_list, test_list = get_train_test_split_lists(images_info_df=images_info_df)

    # moving images/annotation files
    spacer()
    print('copying files to respective folders...')
    copy_images_and_annotations(images_folder_path=images_folder_path,
                                annotations_folder_path=annotations_folder_path,
                                output_folder_path=output_folder_path,
                                train_list=train_list,
                                test_list=test_list)

######################################################################
# defining main function


def main():
    """
    Gets arguments from cli and runs main code.
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
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

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
