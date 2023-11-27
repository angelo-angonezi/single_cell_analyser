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
from pandas import Series
from os.path import exists
from cv2 import INTER_AREA
from pandas import read_csv
from pandas import DataFrame
from cv2 import drawContours
from numpy import add as np_add
from numpy import count_nonzero
from cv2 import IMREAD_GRAYSCALE
from shutil import copy as sh_copy
from cv2 import resize as cv_resize
from numpy import zeros as np_zeroes
from tensorflow import test as tf_test
from scipy.optimize import linear_sum_assignment
from keras.utils import image_dataset_from_directory

# preventing "SettingWithoutCopyWarning" messages
pd.options.mode.chained_assignment = None  # default='warn'

######################################################################
# defining global variables

# image constants
IMAGE_WIDTH = 1408
IMAGE_HEIGHT = 1040
IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT
IMAGE_SIZE = (512, 512)

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


def get_time_str(time_in_seconds: int) -> str:
    """
    Given a time in seconds, returns time in
    adequate format (seconds, minutes or hours).
    :param time_in_seconds: Integer. Represents a time in seconds.
    :return: String. Represents a time (in seconds, minutes or hours).
    """
    # checking whether seconds > 60
    if time_in_seconds >= 60:

        # converting time to minutes
        time_in_minutes = time_in_seconds / 60

        # checking whether minutes > 60
        if time_in_minutes >= 60:

            # converting time to hours
            time_in_hours = time_in_minutes / 60

            # defining time string based on hours
            defined_time = round(time_in_hours)
            time_string = f'{defined_time}h'

        else:

            # defining time string based on minutes
            defined_time = round(time_in_minutes)
            time_string = f'{defined_time}m'

    else:

        # defining time string based on seconds
        defined_time = round(time_in_seconds)
        time_string = f'{defined_time}s'

    # returning time string
    return time_string


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


def simple_hungarian_algorithm(cost_matrix: ndarray) -> tuple:
    """
    Given a cost matrix, applies simple hungarian algorithm
    to establish the best relation between cells.
    :param cost_matrix: ndarray. Represents a cost matrix.
    :return: Tuple. Represents association between cells.
    """
    # assigning relations between cells
    lin_sum_assignment = linear_sum_assignment(cost_matrix)

    # getting assigned cells
    lines = lin_sum_assignment[0].tolist()
    columns = lin_sum_assignment[1].tolist()
    assignments = (lines, columns)

    # returning cells assignments
    return assignments


def add_treatment_col(df: DataFrame,
                      treatment_dict: dict
                      ) -> None:
    """
    Given an analysis data frame, and
    :param df: DataFrame. Represents an analysis data frame.
    :param treatment_dict: Dictionary. Represents a treatment dictionary.
    :return: None.
    """
    # defining treatment col string
    treatment_col = 'Treatment'

    # adding placeholder value for treatment col
    df[treatment_col] = df['Well']

    # defining replacement dict
    replacement_dict = {treatment_col: treatment_dict}

    # updating treatment col values based on treatment dict
    df.replace(replacement_dict,
               inplace=True)


def create_analysis_df(fornma_file_path: str,
                       image_name_col: str,
                       treatment_dict: dict
                       ) -> DataFrame:
    """
    Given a path to a fornma output file,
    returns a data frame, with added/removed
    columns for cell count analysis.
    :param fornma_file_path: String. Represents a path to a fornma output file.
    :param image_name_col: String. Represents a column name.
    :param treatment_dict: Dictionary. Represents a treatment dictionary.
    return: DataFrame. Represents an analysis data frame.
    """
    # reading input file
    fornma_df = read_csv(fornma_file_path)

    # adding new columns based on image extension
    extension_col_split = fornma_df[image_name_col].str.split('.', expand=True)
    new_cols = ['Image_name_no_extension', 'Extension']
    fornma_df[new_cols] = extension_col_split

    # adding new columns based on image name
    extension_col_split = fornma_df['Image_name_no_extension'].str.split('_', expand=True)
    datetime = extension_col_split[9] + extension_col_split[10]
    print(datetime)
    months = [f[5:7] for f in datetime]
    wells = extension_col_split[7]
    fornma_df['Month'] = months
    fornma_df['Well'] = wells
    fornma_df['Datetime'] = datetime
    # new_cols = ['Experiment', 'Well', 'Field', 'Datetime']
    # fornma_df[new_cols] = extension_col_split

    # defining cols to be kept in final df
    cols_to_keep = ['Cell',
                    'Well',
                    'Field',
                    'Datetime']

    '''
    # defining rows to be kept in final df
    all_wells = get_col_set_list(df=fornma_df,
                                 col='Well')
    wells_to_keep = list(treatment_dict.keys())
    wells_to_drop = [well
                     for well
                     in all_wells
                     if well
                     not in wells_to_keep]

    # dropping unrequired columns
    drop_unrequired_cols(df=fornma_df,
                         cols_to_keep=cols_to_keep)

    # dropping unrequired rows
    drop_wells_rows(df=fornma_df,
                    wells=wells_to_drop)
    '''

    # adding treatment column
    add_treatment_col(df=fornma_df,
                      treatment_dict=treatment_dict)

    # returning fornma_df
    return fornma_df


def get_analysis_df(fornma_file_path: str,
                    image_name_col: str,
                    output_folder: str,
                    treatment_dict: dict
                    ) -> DataFrame:
    """
    Returns analysis data frame built
    upon given fornma output file.
    Checks if analysis df is already
    in output folder, and creates/saves
    analysis df should it be non-existent.
    :param fornma_file_path: String. Represents a path to a fornma output file.
    :param image_name_col: String. Represents a column name.
    :param output_folder: String. Represents a path to a folder.
    :param treatment_dict: Dictionary. Represents a treatment dictionary.
    return: DataFrame. Represents an analysis data frame.
    """
    # defining csv output path
    save_name = 'analysis_df.csv'
    save_path = join(output_folder,
                     save_name)

    # defining placeholder value for analysis_df
    analysis_df = None

    # checking if csv output already exists
    if exists(save_path):

        # reading already existent data frame
        print('reading already existent data frame...')
        analysis_df = read_csv(save_path)

    # if output csv does not already exist
    else:

        # creating analysis_df
        print('creating analysis df...')
        analysis_df = create_analysis_df(fornma_file_path=fornma_file_path,
                                         image_name_col=image_name_col,
                                         treatment_dict=treatment_dict)

        # saving output csv
        print('saving analysis df...')
        analysis_df.to_csv(save_path,
                           index=False)

    # returning analysis_df
    return analysis_df


def get_blank_image(width: int,
                    height: int
                    ) -> ndarray:
    """
    Given an image width/height, returns
    numpy array of given dimension
    filled with zeroes.
    :param width: Integer. Represents an image width.
    :param height: Integer. Represents an image height.
    :return: ndarray. Represents an image.
    """
    # defining matrix shape
    shape = (height, width)

    # creating blank matrix
    blank_matrix = np_zeroes(shape=shape)

    # returning blank matrix
    return blank_matrix


def get_iou(mask_a: ndarray,
            mask_b: ndarray
            ) -> float:
    """
    Given two pixel masks, representing
    detected/annotated OBBs, returns IoU.
    :param mask_a: ndarray. Represents a pixel mask.
    :param mask_b: ndarray. Represents a pixel mask.
    :return: Float. Represents an IoU value.
    """
    # adding arrays
    final_array = np_add(mask_a, mask_b)

    # counting "1" pixels (just one of the masks cover)
    one_count = count_nonzero(final_array == 1)

    # counting "2" pixels (= intersection -> both masks cover)
    two_count = count_nonzero(final_array == 2)

    # getting intersection
    intersection = two_count

    # getting union
    union = one_count + two_count

    # calculating IoU (Intersection over Union)
    iou_value = intersection / union

    # returning IoU
    return iou_value


def get_pixel_mask(row_data: Series,
                   style: str
                   ) -> ndarray:
    """
    Given an open image, and coordinates for OBB,
    returns image with respective style overlay.
    :param row_data: Series. Represents OBB coords data.
    :param style: String. Represents an overlay style.
    :return: ndarray. Represents base image with mask overlay.
    """
    # extracting coords from row data
    cx = int(row_data['cx'])
    cy = int(row_data['cy'])
    width = float(row_data['width'])
    height = float(row_data['height'])
    angle = float(row_data['angle'])

    # defining color (same for all styles)
    color = (1,)

    # defining base image
    base_img = get_blank_image(width=IMAGE_WIDTH,
                               height=IMAGE_HEIGHT)

    # checking mask style
    if style == 'ellipse':

        # adding elliptical mask
        draw_ellipse(open_img=base_img,
                     cx=cx,
                     cy=cy,
                     width=width,
                     height=height,
                     angle=angle,
                     color=color,
                     thickness=-1)

    elif style == 'circle':

        # getting radius
        radius = (width + height) / 2
        radius = int(radius)

        # adding circular mask
        draw_circle(open_img=base_img,
                    cx=cx,
                    cy=cy,
                    radius=radius,
                    color=color,
                    thickness=-1)

    elif style == 'rectangle':

        # adding rectangular mask
        draw_rectangle(open_img=base_img,
                       cx=cx,
                       cy=cy,
                       width=width,
                       height=height,
                       angle=angle,
                       color=color,
                       thickness=-1)

    # returning modified image
    return base_img


def get_mask_area(row_data: Series,
                  style: str
                  ) -> int:
    """
    Given an open image, and coordinates for OBB,
    returns image with respective style overlay.
    :param row_data: Series. Represents OBB coords data.
    :param style: String. Represents an overlay style.
    :return: Integer. Represents mask area.
    """
    # getting current pixel mask
    current_mask = get_pixel_mask(row_data=row_data,
                                  style=style)

    # counting "1" pixels (== area occupied by mask)
    mask_area = count_nonzero(current_mask == 1)

    # returning mask area
    return mask_area


def add_area_col(df: DataFrame,
                 style: str
                 ) -> None:
    """
    Given an image data frame, adds
    area column, based on pixel masks
    created according to given style.
    """
    # TODO: check whether this function will be used later on
    # defining area col
    area_col = 'area'

    # defining placeholder value for area col values
    df[area_col] = None

    # getting df rows
    df_rows = df.iterrows()

    # iterating over df rows
    for row_index, row_data in df_rows:

        # getting current row mask area
        current_mask_area = get_mask_area(row_data=row_data,
                                          style=style)

        # updating current row area
        df.at[row_index, area_col] = current_mask_area


def get_image_confluence(df: DataFrame,
                         style: str
                         ) -> float:
    """
    Given an image df, returns given image confluence
    (overlays all detections with given style and counts
    returns non-zero pixels divided by image area).
    """
    # defining base image
    base_img = get_blank_image(width=IMAGE_WIDTH,
                               height=IMAGE_HEIGHT)

    # getting df rows
    df_rows = df.iterrows()

    # iterating over df rows
    for row_index, row_data in df_rows:

        # getting current row pixel mask
        current_mask = get_pixel_mask(row_data=row_data,
                                      style=style)

        # overlaying current mask on base img
        base_img = np_add(base_img, current_mask)

    # getting non-zero pixel count
    non_zeroes_count = count_nonzero(base_img != 0)

    # getting confluence
    confluence = non_zeroes_count / IMAGE_AREA

    # returning confluence
    return confluence


def get_data_split(splits_folder: str,
                   split: str,
                   batch_size: int
                   ):
    """
    Given a path to a folder and a split name,
    returns given data split as tensorflow
    data set.
    """
    # getting train/val/test paths
    data_path = join(splits_folder,
                     split)

    # loading data
    print(f'getting {split} data...')
    # TODO: remove later
    if split == 'test':
        data_path = './/data//nucleus_detection//ImagesFilter//splits//splits_balanced//test'
    print(f'loading data from folder "{data_path}"...')
    split_data = image_dataset_from_directory(directory=data_path,
                                              labels='inferred',
                                              label_mode='binary',
                                              class_names=['excluded', 'included'],
                                              color_mode='rgb',
                                              batch_size=batch_size,
                                              image_size=IMAGE_SIZE,
                                              shuffle=True)

    # returning data
    return split_data


def normalize_data(data):
    """
    Given a tensorflow image data set,
    returns normalized data set so that
    image pixel values range 0-1, instead
    of 0-255.
    """
    # normalizing data
    normalized_data = data.map(lambda x, y: (x / 255, y))

    # returning normalized data
    return normalized_data


def is_using_gpu() -> bool:
    """
    Checks available GPUs and returns
    True if GPU exists and is being used,
    and False otherwise.
    """
    # getting available GPUs
    available_gpus = tf_test.gpu_device_name()

    # checking available GPUs
    if available_gpus:

        # returning True if at least one available
        return True

    # returning False, if none available
    return False


def resize_image(open_image: ndarray,
                 image_size: tuple
                 ) -> ndarray:
    """
    Given an open image, returns resized
    image, based on given image size tuple
    (height, width).
    """
    # getting resized image
    resized_image = cv_resize(open_image,
                              image_size,
                              interpolation=INTER_AREA)

    # returning resized image
    return resized_image

######################################################################
# end of current module
