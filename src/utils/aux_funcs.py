# auxiliary functions module

# Code destined to storing auxiliary
# functions to main module.

######################################################################
# importing required libraries

from os import mkdir
from os import listdir
from sys import stdout
from os.path import join
from pandas import concat
from os.path import exists
from pandas import read_csv
from pandas import DataFrame
from shutil import copy as sh_copy

######################################################################
# defining auxiliary functions


def spacer(char: str = '_',
           reps: int = 50
           ) -> None:
    """
    Given a char and a number of reps,
    prints a "spacer" string assembled
    by multiplying char by reps.
    :param char: String. Represents a character to be used
    as basis for spacer.
    :param reps: Integer. Represents number of character's
    repetitions used in spacer.
    :return: None.
    """
    # defining spacer string
    spacer_str = char * reps

    # printing spacer string
    print(spacer_str)


def flush_string(string: str) -> None:
    """
    Given a string, writes and flushes it in the console using
    sys library, and resets cursor to the start of the line.
    (writes N backspaces at the end of line, where N = len(string)).
    :param string: String. Represents a message to be written in the console.
    :return: None.
    """
    # getting string length
    string_len = len(string)

    # creating backspace line
    backspace_line = '\b' * string_len

    # writing string
    stdout.write(string)

    # flushing console
    stdout.flush()

    # resetting cursor to start of the line
    stdout.write(backspace_line)


def flush_or_print(string: str,
                   index: int,
                   total: int
                   ) -> None:
    """
    Given a string, prints string if index
    is equal to total, and flushes it on console
    otherwise.
    !(useful for progress tracking/progress bars)!
    :param string: String. Represents a string to be printed on console.
    :param index: Integer. Represents an iterable's index.
    :param total: Integer. Represents an iterable's total.
    :return: None.
    """
    # checking whether index is last

    # if current element is last
    if index == total:

        # printing string
        print(string)

    # if current element is not last
    else:

        # flushing string
        flush_string(string)


def get_specific_files_in_folder(path_to_folder: str,
                                 extension: str
                                 ) -> list:
    """
    Given a path to a folder, returns a list containing
    all files in folder that match given extension.
    :param path_to_folder: String. Represents a path to a folder.
    :param extension: String. Represents a specific file extension.
    :return: List[str]. Represents all files that match extension in given folder.
    """
    # getting all files in folder
    all_files_in_folder = listdir(path_to_folder)

    # getting specific files
    files_in_dir = [file                          # getting file
                    for file                      # iterating over files
                    in all_files_in_folder        # in input folder
                    if file.endswith(extension)]  # only if file matches given extension

    # sorting list
    files_in_dir = sorted(files_in_dir)

    # returning list
    return files_in_dir


def get_data_from_consolidated_df(consolidated_df_file_path: str) -> DataFrame:
    """
    Given a path to a consolidated dataframe,
    returns processed dataframe.
    :param consolidated_df_file_path: String. Represents a path to a file.
    :return: DataFrame. Represents data contained in input file.
    """
    # reading df from file path
    consolidated_df = read_csv(consolidated_df_file_path)

    # returning df
    return consolidated_df


def get_obbs_from_df(df: DataFrame) -> list:
    """
    Given a detections data frame, returns
    a list of detected OBBs info, in following
    format:
    [(cx, cy, width, height, angle), ...]
    :param df: DataFrame. Represents detections data frame.
    :return: List. Represents OBBs in given data frame.
    """
    # getting cx values
    cxs = df['cx']
    cxs = cxs.astype(int)

    # getting cy values
    cys = df['cy']
    cys = cys.astype(int)

    # getting width values
    widths = df['width']
    widths = widths.astype(int)

    # getting height values
    heights = df['height']
    heights = heights.astype(int)

    # getting angle values
    angles = df['angle']

    # creating zip list
    centroids_list = [(cx, cy, width, height, angle)
                      for cx, cy, width, height, angle
                      in zip(cxs, cys, widths, heights, angles)]

    # returning centroids list
    return centroids_list


def create_folder(folder_path: str) -> None:
    """
    Given a path to a folder, checks folder
    existence, and creates folder should it
    be non-existent.
    :param folder_path: String. Represents a path to a folder.
    :return: None.
    """
    # checking if folder exists
    if exists(folder_path):

        # does nothing (return None)
        return None

    # if it does not exist
    else:

        # creating folder
        mkdir(folder_path)


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
        create_folder(subfolder_path)


def copy_multiple_files(src_folder_path: str,
                        dst_folder_path: str,
                        files_list: list,
                        file_extension: str
                        ) -> None:
    """
    Given a path to source and destination folders,
    and a list of files present in source folder,
    copies files to destination folder.
    :param src_folder_path: String. Represents a path to a folder.
    :param dst_folder_path: String. Represents a path to a folder.
    :param files_list: List. Represents file names.
    :param file_extension: String. Represents file extension.
    :return: None.
    """
    # getting files number
    files_num = len(files_list)

    # iterating over files
    for file_index, file_name in enumerate(files_list, 1):

        # getting file src/dst paths
        file_path = f'{file_name}{file_extension}'
        src_path = join(src_folder_path, file_path)
        dst_path = join(dst_folder_path, file_path)

        # printing execution message
        progress_ratio = file_index / files_num
        progress_percentage = progress_ratio * 100
        progress_percentage_round = round(progress_percentage)
        f_string = f'copying file {file_index} of {files_num} ({progress_percentage_round}%)'
        flush_or_print(string=f_string,
                       index=file_index,
                       total=files_num)

        # copying file from src to dst folder
        sh_copy(src=src_path,
                dst=dst_path)


def print_execution_parameters(params_dict: dict) -> None:
    """
    Given a list of execution parameters,
    prints given parameters on console,
    such as:
    '''
    --Execution parameters--
    input_folder: /home/angelo/Desktop/ml_temp/imgs/
    output_folder: /home/angelo/Desktop/ml_temp/overlays/
    '''
    :param params_dict: Dictionary. Represents execution parameters names and values.
    :return: None.
    """
    # defining base params_string
    params_string = f'--Execution parameters--'

    # iterating over parameters in params_list
    for dict_element in params_dict.items():

        # getting params key/value
        param_key, param_value = dict_element

        # adding 'Enter' to params_string
        params_string += '\n'

        # getting current param string
        current_param_string = f'{param_key}: {param_value}'

        # appending current param string to params_string
        params_string += current_param_string

    # printing final params_string
    spacer()
    print(params_string)
    spacer()


def enter_to_continue():
    """
    Waits for user input ("Enter")
    and once press, continues to run code.
    """
    # defining enter_string
    enter_string = f'press "Enter" to continue'

    # waiting for user input
    input(enter_string)


def get_merged_detection_annotation_df(detections_df_path: str,
                                       annotations_df_path: str or None
                                       ) -> DataFrame:
    """
    Given a path to detections df and annotations df,
    returns merged df, containing new column "evaluator",
    representing detection/annotation info.
    :param detections_df_path: String. Represents a path to a file.
    :param annotations_df_path: String. Represents a path to a file.
    :return: DataFrame. Represents merged detection/annotation data.
    """
    # defining placeholder value for dfs_list
    dfs_list = []

    # reading detections file
    print('reading detections file...')
    detections_df = read_csv(detections_df_path)

    # adding evaluator constant column
    detections_df['evaluator'] = 'model'

    # adding detections df to dfs_list
    dfs_list.append(detections_df)

    # checking ground_truth_file_path
    if annotations_df_path is not None:

        # reading gt file
        print('reading ground-truth file...')
        ground_truth_df = read_csv(annotations_df_path)

        # adding evaluator constant column
        ground_truth_df['evaluator'] = 'fornma'

        # adding annotations df to dfs_list
        dfs_list.append(ground_truth_df)

    # concatenating dfs in dfs_list
    merged_df = concat(dfs_list)

    # returning merged df
    return merged_df

######################################################################
# end of current module
