# auxiliary functions module

# Code destined to storing auxiliary
# functions to main module.

######################################################################
# imports

# importing required libraries
import pandas as pd
from os import mkdir
from time import time
from cv2 import circle
from numpy import intp
from os import listdir
from cv2 import imread
from sys import stdout
from cv2 import ellipse
from os.path import join
from numpy import ndarray
from cv2 import boxPoints
from pandas import concat
from os.path import exists
from pandas import read_csv
from cv2 import drawContours
from pandas import DataFrame
from cv2 import IMREAD_GRAYSCALE
from shutil import copy as sh_copy
from scipy.optimize import linear_sum_assignment

# preventing "SettingWithoutCopyWarning" messages
pd.options.mode.chained_assignment = None  # default='warn'

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


def print_progress_message(base_string: str,
                           index: int,
                           total: int
                           ) -> None:
    """
    Given a base string (containing keywords #INDEX#
    and #TOTAL#) an index and a total (integers),
    prints execution message, substituting #INDEX#
    and #TOTAL# keywords by respective integers.
    !!!Useful for FOR loops execution messages!!!
    :param base_string: String. Represents a base string.
    :param index: Integer. Represents an execution index.
    :param total: Integer. Represents iteration maximum value.
    :return: None.
    """
    # getting percentage progress
    progress_ratio = index / total
    progress_percentage = progress_ratio * 100
    progress_percentage_round = round(progress_percentage)

    # assembling progress string
    progress_string = base_string.replace('#INDEX#', str(index))
    progress_string = progress_string.replace('#TOTAL#', str(total))
    progress_string += '...'
    progress_string += f' ({progress_percentage_round}%)'

    # showing progress message
    flush_or_print(string=progress_string,
                   index=index,
                   total=total)


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


def get_current_time() -> int:
    """
    Gets current UTC time, in seconds.
    """
    # getting current time
    current_time = time()

    # getting seconds
    current_seconds = int(current_time)

    # returning current time in seconds
    return current_seconds


def get_time_elapsed(start_time: int,
                     current_time: int
                     ) -> int:
    """
    Given two UTC times, returns
    difference between times, in
    seconds.
    """
    # getting time diff
    time_diff = current_time - start_time

    # returning time diff
    return time_diff


def get_etc(time_elapsed: int,
            current_iteration: int,
            iterations_total: int
            ) -> int:
    """
    Given the time elapse for achieving
    current iteration and an iterations
    total, returns estimated time of
    completion (ETC), in seconds.
    """
    # getting iterations to go
    iterations_to_go = iterations_total - current_iteration

    # calculating estimated time of completion
    etc = iterations_to_go * time_elapsed / current_iteration

    # converting estimated time of completion to int
    etc = int(etc)

    # returning estimated time of completion
    return etc


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

    # getting class values
    cell_classes = df['class']
    cell_classes = cell_classes.astype(str)

    # creating zip list
    centroids_list = [(cx, cy, width, height, angle, cell_class)
                      for cx, cy, width, height, angle, cell_class
                      in zip(cxs, cys, widths, heights, angles, cell_classes)]

    # returning centroids list
    return centroids_list


def drop_unrequired_cols(df: DataFrame,
                         cols_to_keep: list
                         ) -> None:
    """
    Given a data frame, and a list of
    columns to be kept in final df,
    returns updated df, in which columns
    which are not in given list have been
    dropped.
    :param df: DataFrame. Represents a data frame.
    :param cols_to_keep: List. Represents column names.
    :return: None.
    """
    # getting current cols
    cols = df.columns

    # getting cols to drop
    cols_to_drop = [col
                    for col
                    in cols
                    if col
                    not in cols_to_keep]

    # iterating over cols_to_drop
    for col in cols_to_drop:

        # dropping current col
        df.drop(col,
                axis=1,
                inplace=True)


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
        f_string = f'copying file #INDEX# of #TOTAL#'
        print_progress_message(base_string=f_string,
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


def get_merged_detection_annotation_df(detections_df_path: str or None,
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

    # checking if there's at least one file path
    both_none = (detections_df_path is None) and (annotations_df_path is None)
    if both_none:

        # printing execution message
        f_string = 'No input file containing detections provided\n'
        f_string += 'Please, check input and try again.'
        print(f_string)
        exit()

    # checking detections_df_path
    if detections_df_path is not None:

        # reading detections file
        print('reading detections file...')
        detections_df = read_csv(detections_df_path)

        # adding evaluator constant column
        detections_df['evaluator'] = 'model'

        # adding annotations df to dfs_list
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
    print('merging dfs...')
    merged_df = concat(dfs_list)

    # returning merged df
    return merged_df


def get_area(width: float,
             height: float
             ) -> float:
    """
    Given width and height values, returns
    area (multiplication of width and height).
    :param width: Float. Represents OBB width.
    :param height: Float. Represents OBB height.
    :return: Float. Represents axis ratio.
    """
    # calculating area
    area = width * height

    # returning area
    return area


def get_axis_ratio(width: float,
                   height: float
                   ) -> float:
    """
    Given width and height values, checks which one
    is larger, and returns ratio between longer
    and shorter axis.
    :param width: Float. Represents OBB width.
    :param height: Float. Represents OBB height.
    :return: Float. Represents axis ratio.
    """
    # defining long and short axis based on width/height values
    long_axis = width if width > height else height
    short_axis = width if width < height else height

    # calculating axis_ratio
    axis_ratio = long_axis / short_axis

    # returning axis_ratio
    return axis_ratio


def add_nma_col(df: DataFrame,
                col_name: str
                ) -> None:
    """
    Given a merged detections/annotations data frame,
    adds 'area' or 'axis_ratio' column, calculated by
    multiplying/dividing width/height cols.
    :param df: DataFrame. Represents merged detections/annotations data.
    :param col_name: String. Represents a column name.
    :return: None.
    """
    # adding placeholder column to df
    df[col_name] = None

    # getting df rows
    df_rows = df.iterrows()

    # getting rows total
    rows_num = len(df)

    # defining placeholder value for current_row_index
    current_row_index = 1

    # defining progress base string
    progress_base_string = f'adding {col_name} column to row #INDEX# of #TOTAL#'

    # iterating over df rows
    for row_index, row_data in df_rows:

        # printing execution message
        print_progress_message(base_string=progress_base_string,
                               index=current_row_index,
                               total=rows_num)

        # defining placeholder value for current_row_value
        current_row_value = None

        # getting current row width/height data
        current_width = row_data['width']
        current_height = row_data['height']

        # updating current_row_value

        if col_name == 'area':

            # getting current area
            current_row_value = get_area(width=current_width,
                                         height=current_height)

        elif col_name == 'axis_ratio':

            # getting current axis ratio
            current_row_value = get_axis_ratio(width=current_width,
                                               height=current_height)

        else:

            # printing error message
            e_string = f'{col_name} undefined.'
            e_string += f'Valid inputs are : "area" and "axis_ratio"'
            e_string += f'Please, check and try again.'
            print(e_string)
            exit()

        # updating current line area/axis_ratio value
        df.at[row_index, col_name] = current_row_value

        # updating row index
        current_row_index += 1


def add_date_col(df: DataFrame) -> None:
    """
    Given a merged detections/annotations data frame,
    adds 'date' column, to enable grouping by time stamp.
    :param df: DataFrame. Represents merged detections/annotations data.
    :return: None.
    """
    # defining column name
    col_name = 'date'

    # adding placeholder column to df
    df[col_name] = None

    # getting df rows
    df_rows = df.iterrows()

    # getting rows total
    rows_num = len(df)

    # defining placeholder value for current_row_index
    current_row_index = 1

    # defining progress base string
    progress_base_string = f'adding {col_name} column to row #INDEX# of #TOTAL#'

    # iterating over df rows
    for row_index, row_data in df_rows:

        # printing execution message
        print_progress_message(base_string=progress_base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row date data
        current_img_name = row_data['img_file_name']
        current_img_name_split = current_img_name.split('_')
        current_img_date = current_img_name_split[-1]

        # updating current row value
        df.at[row_index, col_name] = current_img_date

        # updating row index
        current_row_index += 1


def add_treatment_col_fer(df: DataFrame) -> None:
    """
    Given a merged detections/annotations data frame,
    adds 'treatment' column, obtained by file name.
    :param df: DataFrame. Represents merged detections/annotations data.
    :return: None.
    """
    # adding treatment placeholder column to df
    df['treatment'] = None

    # getting df rows
    df_rows = df.iterrows()

    # iterating over df rows
    for row_index, row_data in df_rows:

        # getting current row treatment data
        img_file_name = row_data['img_file_name']
        img_file_name_split = img_file_name.split('_')
        treatment_col = img_file_name_split[1]
        treatment_str = treatment_col[0]

        # defining current treatment
        current_treatment = 'CTR' if treatment_str == 'B' else 'ATF6'

        # updating current line axis ratio value
        df.at[row_index, 'treatment'] = current_treatment


def add_treatment_col_debs(df: DataFrame) -> None:
    """
    Given a merged detections/annotations data frame,
    adds 'treatment' column, obtained by file name.
    :param df: DataFrame. Represents merged detections/annotations data.
    :return: None.
    """
    # TODO: adjust this function to add TMZ/CTR column
    # adding treatment placeholder column to df
    df['treatment'] = None

    # getting df rows
    df_rows = df.iterrows()

    # iterating over df rows
    for row_index, row_data in df_rows:

        # getting current row treatment data
        img_file_name = row_data['img_file_name']
        img_file_name_split = img_file_name.split('_')
        treatment_col = img_file_name_split[1]
        treatment_str = treatment_col[0]

        # defining current treatment
        current_treatment = 'CTR' if treatment_str == 'B' else 'TTO'

        # updating current line axis ratio value
        df.at[row_index, 'treatment'] = current_treatment


def add_treatment_col_daph(df: DataFrame,
                           data_format: str
                           ) -> None:
    """
    Given a merged detections/annotations data frame,
    adds 'treatment' column, obtained by file name.
    :param df: DataFrame. Represents merged detections/annotations data.
    :param data_format: DataFrame. Represents merged detections/annotations data.
    :return: None.
    """
    # adding treatment placeholder column to df
    df['treatment'] = None

    # defining file_name_col based on data_type string
    file_name_col = 'Image_name_red' if data_format == 'fornma' else 'img_file_name'

    # defining treatment_dict
    treatment_dict = {'B1': 'CTR',
                      'C1': 'CTR',
                      'B4': 'TMZ_10uM',
                      'C4': 'TMZ_10uM',
                      'B5': 'TMZ_50uM',
                      'C5': 'TMZ_50uM',
                      'B6': 'TMZ_100uM',
                      'C6': 'TMZ_100uM'}

    # getting df rows
    df_rows = df.iterrows()

    # getting rows total
    rows_num = len(df)

    # defining placeholder value for current_row_index
    current_row_index = 1

    # defining progress base string
    progress_base_string = f'adding treatment col to row #INDEX# of #TOTAL#'

    # iterating over df rows
    for row_index, row_data in df_rows:

        # printing execution message
        print_progress_message(base_string=progress_base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row treatment data
        img_file_name = row_data[file_name_col]
        img_file_name_split = img_file_name.split('_')
        treatment_col = img_file_name_split[1]

        # defining current treatment
        current_treatment = treatment_dict[treatment_col]

        # updating current line axis ratio value
        df.at[row_index, 'treatment'] = current_treatment

        # updating row index
        current_row_index += 1


def get_crop_pixels(crop_path: str) -> ndarray:
    """
    Given a path to a crop, returns
    crops pixels, in a linearized array.
    :param crop_path: String. Represents a path to a crop.
    :return: ndarray. Represents a crop's pixels.
    """
    # opening image
    open_crop = imread(crop_path,
                       IMREAD_GRAYSCALE)

    # linearizing pixels
    linearized_pixels = open_crop.flatten()

    # returning crop's linearized pixels
    return linearized_pixels


def draw_rectangle(open_img: ndarray,
                   cx: float,
                   cy: float,
                   width: float,
                   height: float,
                   angle: float,
                   color: tuple,
                   thickness: int
                   ) -> ndarray:
    """
    Given an open image, and coordinates for OBB,
    returns image with OBB rectangular overlay.
    """
    # get the corner points
    box = boxPoints(((cx, cy),
                     (width, height),
                     angle))

    # converting corners format
    box = intp(box)

    # drawing rectangle on image
    drawContours(open_img,
                 [box],
                 -1,
                 color,
                 thickness=thickness)

    # returning modified image
    return open_img


def draw_circle(open_img: ndarray,
                cx: float,
                cy: float,
                radius: float,
                color: tuple,
                thickness: int
                ) -> ndarray:
    """
    Given an open image, and coordinates for OBB,
    returns image with OBB circular overlay.
    """
    # drawing circle on image
    circle(open_img,
           (cx, cy),
           radius,
           color,
           thickness=thickness)

    # returning modified image
    return open_img


def draw_ellipse(open_img: ndarray,
                 cx: float,
                 cy: float,
                 width: float,
                 height: float,
                 angle: float,
                 color: tuple,
                 thickness: int
                 ) -> ndarray:
    """
    Given an open image, and coordinates for OBB,
    returns image with OBB elliptical overlay.
    """
    # dividing axes length by two (cv2.ellipse takes the radius)
    width = width / 2
    height = height / 2

    # defining center/axes
    center = (int(cx), int(cy))
    axes = (int(width), int(height))

    # drawing ellipse on image
    ellipse(img=open_img,
            center=center,
            axes=axes,
            angle=angle,
            color=color,
            thickness=thickness,
            startAngle=0,
            endAngle=360)

    # returning modified image
    return open_img


def simple_hungarian_algorithm(cost_matrix: ndarray) -> list:
    """
    Given a cost matrix, applies simple hungarian algorithm
    to establish the best relation between cells.
    :param cost_matrix: ndarray. Represents a cost matrix.
    :return: List. Represents association between cells.
    """
    # assigning relations between cells
    lin_sum_assignment = linear_sum_assignment(cost_matrix)

    # getting assigned cells
    lines = lin_sum_assignment[0].tolist()
    columns = lin_sum_assignment[1].tolist()
    assignments = [lines, columns]

    # returning cells assignments
    return assignments

######################################################################
# end of current module
