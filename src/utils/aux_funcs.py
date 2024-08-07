# auxiliary functions module

# Code destined to storing auxiliary
# functions to main module.

######################################################################
# imports

# importing required libraries
from os import sep
import pandas as pd
from os import mkdir
from cv2 import flip
from time import time
from cv2 import rotate
from cv2 import circle
from cv2 import imshow
from numpy import intp
from os import listdir
from os import environ
from cv2 import imread
from sys import stdout
from cv2 import waitKey
from cv2 import imwrite
from cv2 import ellipse
from os.path import join
from cv2 import cvtColor
from numpy import ndarray
from cv2 import boxPoints
from pandas import concat
from pandas import Series
from os.path import exists
from cv2 import ROTATE_180
from cv2 import INTER_AREA
from cv2 import contourArea
from pandas import read_csv
from seaborn import lineplot
from cv2 import drawContours
from pandas import DataFrame
from cv2 import boundingRect
from cv2 import findContours
from cv2 import RETR_EXTERNAL
from cv2 import COLOR_GRAY2BGR
from scipy.stats import levene
from cv2 import convertScaleAbs
from numpy import add as np_add
from numpy import count_nonzero
from scipy.stats import kruskal
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from cv2 import IMREAD_GRAYSCALE
from cv2 import destroyAllWindows
from scipy.stats import ttest_rel
from cv2 import CHAIN_APPROX_NONE
from shutil import copy as sh_copy
from numpy import where as np_where
from cv2 import resize as cv_resize
from scipy.stats import mannwhitneyu
from matplotlib import pyplot as plt
from numpy import zeros as np_zeroes
from tensorflow import test as tf_test
from scipy.optimize import linear_sum_assignment
from keras.utils import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from keras.applications.resnet import preprocess_input as resnet_preprocess_input

# preventing "SettingWithoutCopyWarning" messages
pd.options.mode.chained_assignment = None  # default='warn'

# setting tensorflow warnings off
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

######################################################################
# defining global variables

# image constants
IMAGE_WIDTH = 1408
IMAGE_HEIGHT = 1040
IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT
# IMAGE_SIZE = (50, 50)
# IMAGE_SIZE = (64, 64)
IMAGE_SIZE = (100, 100)
# IMAGE_SIZE = (224, 224)
# IMAGE_SIZE = (384, 384)
# IMAGE_SIZE = (512, 512)
IMG_HEIGHT, IMG_WIDTH = IMAGE_SIZE
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)  # 3 because it is an RGB image

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
    progress_string += '     '

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


def get_dirs_in_folder(path_to_folder: str) -> list:
    """
    Given a path to a folder, returns
    a list containing all subfolders
    in given folder.
    :param path_to_folder: String. Represents a path to a folder.
    :return: List[str]. Represents all subfolders in given folder.
    """
    # getting all files in folder
    all_files_in_folder = listdir(path_to_folder)

    # getting specific files
    files_in_dir = [file                          # getting file
                    for file                      # iterating over files
                    in all_files_in_folder]       # in input folder

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


def get_test_images_df(df: DataFrame) -> DataFrame:
    """
    Given a merged detections/annotations data frame,
    returns a data frame containing only rows which
    contain at least two annotators (fornma+model).
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # defining group col
    groups = 'img_file_name'

    # grouping df
    df_groups = df.groupby(groups)

    # iterating over df groups
    for img_name, img_group in df_groups:

        # getting current image evaluators num
        evaluators_col = img_group['evaluator']
        evaluators = evaluators_col.unique()
        evaluators_num = len(evaluators)

        # checking whether current image group contains just fornma annotations
        if evaluators_num == 1:

            # skipping current image
            continue

        # appending current image group to dfs list
        dfs_list.append(img_group)

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


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
    open_crop = load_grayscale_img(image_path=crop_path)

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
                 color: tuple or int,
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


def add_experiment_cols(df: DataFrame,
                        file_name_col: str
                        ) -> None:
    """
    Given an analysis (fornma/model/crops)
    data frame, adds experiment related columns
    based on file name, specified by given col.
    """
    # defining col names
    experiment_col = 'experiment'
    well_col = 'well'
    field_col = 'field'
    date_col = 'date'
    time_col = 'time'

    # adding placeholder values to cols
    df[experiment_col] = None
    df[well_col] = None
    df[field_col] = None
    df[date_col] = None
    df[time_col] = None

    # getting df rows
    df_rows = df.iterrows()

    # getting rows num
    rows_num = len(df)

    # defining starter for current row index
    current_row_index = 1

    # iterating over rows
    for row in df_rows:

        # printing progress message
        base_string = f'adding experiment cols (row #INDEX# of #TOTAL#)'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row index/data
        row_index, row_data = row

        # getting current row image name
        current_img_name = row_data[file_name_col]

        # removing current image extension
        current_img_name = current_img_name.replace('.tif', '')
        current_img_name = current_img_name.replace('.jpg', '')
        current_img_name = current_img_name.replace('.png', '')

        # splitting current file name split
        img_split = current_img_name.split('_')

        # getting file name related info
        current_experiment_split = img_split[0:-4]
        current_experiment = '_'.join(current_experiment_split)
        current_well = img_split[-4]
        current_field = img_split[-3]
        current_date = img_split[-2]
        current_time = img_split[-1]

        # updating current row cols
        df.at[row_index, experiment_col] = current_experiment
        df.at[row_index, well_col] = current_well
        df.at[row_index, field_col] = current_field
        df.at[row_index, date_col] = current_date
        df.at[row_index, time_col] = current_time

        # updating current row index
        current_row_index += 1


def get_treatment_dict(treatment_file: str) -> dict:
    """
    Given a path to a file containing
    treatment info, returns a dictionary
    representing well-treatment relationship.
    """
    # defining placeholder value for treatment dict
    treatment_dict = {}

    # checking if treatment file exists
    treatment_file_exists = exists(treatment_file)

    # if it does not exist
    if not treatment_file_exists:

        # returning empty dict
        return treatment_dict

    # reading input file
    with open(treatment_file, 'r') as open_file:

        # getting file lines
        lines = open_file.readlines()

        # iterating over file lines
        for line in lines:

            # removing "enter"
            line = line.replace('\n', '')

            # getting current line treatment key/value
            current_line_split = line.split('=')
            current_key = current_line_split[0]
            current_value = current_line_split[1]

            # assembling new dict element
            new_dict_element = {current_key: current_value}

            # updating treatment dict
            treatment_dict.update(new_dict_element)

    # returning treatment dict
    return treatment_dict


def add_treatment_col(df: DataFrame,
                      treatment_dict: dict
                      ) -> None:
    """
    Given an analysis data frame, and
    a treatment dict, adds treatment col
    to data frame, based on each image well.
    """
    # defining treatment col string
    treatment_col = 'treatment'

    # adding placeholder value for treatment col
    df[treatment_col] = df['well']

    # defining replacement dict
    replacement_dict = {treatment_col: treatment_dict}

    # updating treatment col values based on treatment dict
    df.replace(replacement_dict,
               inplace=True)


def create_analysis_df(base_df_path: str,
                       image_name_col: str,
                       treatment_file: str
                       ) -> DataFrame:
    """
    Given a path to a base data frame
    (crops_info, fornma or model detections),
    adds image name-based columns, such as
    Well, Field, and Treatment, returning
    analysis data frame.
    """
    # reading base df
    analysis_df = read_csv(base_df_path)

    # adding experiment cols
    print('adding experiment cols...')
    add_experiment_cols(df=analysis_df,
                        file_name_col=image_name_col)

    # getting treatment dict
    treatment_dict = get_treatment_dict(treatment_file=treatment_file)

    # adding treatment column
    print('adding treatment col...')
    add_treatment_col(df=analysis_df,
                      treatment_dict=treatment_dict)

    # returning base df
    return analysis_df


def get_analysis_df(base_df_path: str,
                    image_name_col: str,
                    output_folder: str,
                    treatment_file: str
                    ) -> DataFrame:
    """
    Returns analysis data frame built
    upon given fornma output file.
    Checks if analysis df is already
    in output folder, and creates/saves
    analysis df should it be non-existent.
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
        analysis_df = create_analysis_df(base_df_path=base_df_path,
                                         image_name_col=image_name_col,
                                         treatment_file=treatment_file)

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
                   style: str,
                   expansion_ratio: float = 1.0,
                   crop_index: int = 1
                   ) -> ndarray:
    """
    Given an open image, and coordinates for OBB,
    returns image with respective style overlay.
    """
    # extracting coords from row data
    cx = int(row_data['cx'])
    cy = int(row_data['cy'])
    width = float(row_data['width'])
    height = float(row_data['height'])
    angle = float(row_data['angle'])

    # expanding with/height
    width = width * expansion_ratio
    height = height * expansion_ratio

    # defining color (based on crop index)
    color = (crop_index,)

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


def get_segmentation_mask(df: DataFrame,
                          style: str,
                          expansion_ratio: float = 1.0
                          ) -> ndarray:
    """
    Given an image df, adds overlays to all
    detections with given style and returns
    a binary image where zero pixels mean
    background and non-zero mean nuclei.
    """
    # defining base image
    base_img = get_blank_image(width=IMAGE_WIDTH,
                               height=IMAGE_HEIGHT)

    # getting df rows
    df_rows = df.iterrows()

    # defining starter for current_crop_index
    current_crop_index = 1

    # iterating over df rows
    for row_index, row_data in df_rows:

        # getting current row pixel mask
        current_mask = get_pixel_mask(row_data=row_data,
                                      style=style,
                                      expansion_ratio=expansion_ratio,
                                      crop_index=current_crop_index)

        # overlaying current mask on base img
        base_img = np_add(base_img, current_mask)

        # updating current_crop_index
        current_crop_index += 1

    # returning base img
    return base_img


def get_image_confluence(df: DataFrame,
                         style: str
                         ) -> float:
    """
    Given an image df, returns given image confluence
    (overlays all detections with given style and counts
    returns non-zero pixels divided by image area).
    """
    # getting segmentation mask
    segmentation_mask = get_segmentation_mask(df=df,
                                              style=style)

    # getting non-zero pixel count
    non_zeroes_count = count_nonzero(segmentation_mask != 0)

    # getting confluence
    confluence = non_zeroes_count / IMAGE_AREA

    # returning confluence
    return confluence


def get_data_split_from_df(splits_folder: str,
                           extension: str,
                           dataset_df: DataFrame,
                           split: str,
                           batch_size: int,
                           class_mode: str
                           ):
    """
    Given a path to a folder and a split name,
    returns given data split as tensorflow
    data set.
    """
    # printing execution message
    print(f'getting {split} data...')

    # filtering df by current split
    filtered_df = dataset_df[dataset_df['split'] == split]
    rows_num = len(filtered_df)
    rows_range = range(rows_num)

    # adding current folder column
    current_split_folder = splits_folder + sep + split
    folder_col = [current_split_folder for _ in rows_range]
    filtered_df['input_folder'] = folder_col

    # adding crop name with extension col
    name_w_extension_col = filtered_df['crop_name'] + extension
    filtered_df['crop_name_w_extension'] = name_w_extension_col

    # adding file path column
    path_col = filtered_df['input_folder'] + sep + filtered_df['crop_name_w_extension']
    filtered_df['crop_path'] = path_col

    # dropping unrequired columns
    cols_to_keep = ['crop_path', 'class']
    filtered_df = filtered_df[cols_to_keep]

    # creating image data generator object
    # image_generator = ImageDataGenerator(preprocessing_function=vgg_preprocess_input)
    image_generator = ImageDataGenerator(preprocessing_function=resnet_preprocess_input)
    # rescaler = 1 / 255
    # image_generator = ImageDataGenerator(rescale=rescaler)

    # loading data
    print(f'loading data from folder "{current_split_folder}"...')
    split_data = image_generator.flow_from_dataframe(dataframe=filtered_df,
                                                     x_col='crop_path',
                                                     y_col='class',
                                                     target_size=IMAGE_SIZE,
                                                     color_mode='rgb',
                                                     class_mode=class_mode,  # 'raw' for regression, 'binary' for binary classification and 'categorical' for multi-label classification  # noqa
                                                     batch_size=batch_size,
                                                     shuffle=True)

    print(split_data[0])

    # returning data
    return split_data


def get_data_split_from_folder(splits_folder: str,
                               split: str,
                               batch_size: int
                               ):
    """
    Given a path to a folder and a split name,
    returns given data split as tensorflow
    data set.
    """
    # printing execution message
    print(f'getting {split} data...')

    # getting train/val/test paths
    data_path = join(splits_folder,
                     split)

    # loading data
    print(f'loading data from folder "{data_path}"...')
    split_data = image_dataset_from_directory(directory=data_path,
                                              labels='inferred',
                                              label_mode='binary',
                                              class_names=['excluded', 'included'],
                                              color_mode='rgb',
                                              batch_size=batch_size,
                                              image_size=IMAGE_SIZE,
                                              shuffle=True)

    # normalizing data to 0-1 scale
    print('normalizing data...')
    normalized_data = normalize_data(data=split_data)

    # returning normalized data
    return normalized_data


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


def print_gpu_usage() -> None:
    """
    Checks GPU usage and prints
    pretty string on console.
    """
    # getting gpu usage bool
    using_gpu = is_using_gpu()

    # defining base string
    base_string = f'Using GPU: {using_gpu}'

    # printing string
    print(base_string)


def enhance_contrast(image_matrix, bins=256):
    import cv2
    import numpy as np

    img = image_matrix
    # converting to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    result = np.hstack((img, enhanced_img))

    return result


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


def get_experiment_well_df(df: DataFrame,
                           experiment: str,
                           well: str
                           ) -> DataFrame:
    """
    Given a data frame, an experiment name
    and a well, returns df filtered by given
    experiment and well.
    """
    # filtering df by experiment name
    experiment_df = df[df['experiment'] == experiment]

    # filtering df by well
    wells_df = experiment_df[experiment_df['well'] == well]

    # getting row
    row = wells_df.iloc[0]

    # returning filtered df row
    return row


def add_confluence_group_col(df: DataFrame) -> None:
    """
    Docstring.
    """
    # adding confluence percentage col
    try:
        df['confluence_percentage'] = df['confluence']
    except KeyError:
        df['confluence_percentage'] = df['fornma_confluence']

    # getting confluence percentage round values
    df['confluence_percentage_round'] = df['confluence_percentage'].round()

    # getting confluence percentage int values
    df['confluence_percentage_int'] = df['confluence_percentage_round'].astype(int)

    # getting confluence percentage str values
    df['confluence_percentage_str'] = df['confluence_percentage_int'].astype(str)

    # getting confluence group values
    df['confluence_group'] = df['confluence_percentage_str'].replace('0', '<1')


def add_confluence_level_col(df: DataFrame) -> None:
    """
    Docstring.
    """
    # getting confluence percentage mean/std
    conf_mean = df['confluence_percentage'].mean()
    conf_std = df['confluence_percentage'].std()
    print('defining  confluence levels thresholds...')
    print(f'conf_mean: {conf_mean}')
    print(f'conf_std: {conf_std}')

    # calculating confluence threshold
    multiplier = 1.0
    # confluence_lower_threshold = conf_mean - (multiplier * conf_std)
    confluence_lower_threshold = 7
    # confluence_upper_threshold = conf_mean + (multiplier * conf_std)
    confluence_upper_threshold = 14
    print(f'confluence_lower_threshold: {confluence_lower_threshold}')
    print(f'confluence_upper_threshold: {confluence_upper_threshold}')

    # getting confluence percentage col
    confluence_percentage_col = df['confluence_percentage']

    # calculating confluence level for values
    confluence_levels_col = []
    for value in confluence_percentage_col:
        level = None
        if value < confluence_lower_threshold:
            level = 'Low'
        elif value > confluence_upper_threshold:
            level = 'High'
        else:
            level = 'Medium'
        confluence_levels_col.append(level)

    # adding confluence level col
    df['confluence_level'] = confluence_levels_col


def get_base_df(files: list,
                col_name: str
                ) -> DataFrame:
    """
    Given a list of files, returns base
    data frame, used on following analysis.
    """
    # assembling new col
    new_col = {col_name: files}

    # creating data frame
    base_df = DataFrame(new_col)

    # returning base df
    return base_df


def add_file_path_col(df: DataFrame,
                      input_folder: str
                      ) -> None:
    """
    Given a base image names data frame,
    adds file path column, based on given
    input folder.
    """
    # defining col name
    col_name = 'file_path'

    # adding placeholder values to col
    df[col_name] = None

    # getting df rows
    df_rows = df.iterrows()

    # getting rows num
    rows_num = len(df)

    # defining starter for current row index
    current_row_index = 1

    # iterating over rows
    for row in df_rows:

        # printing progress message
        base_string = f'adding file path col (row #INDEX# of #TOTAL#)'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row index/data
        row_index, row_data = row

        # getting current row image name
        file_name = row_data['file_name']

        # getting current row file path
        current_file_path = join(input_folder,
                                 file_name)

        # updating current row col
        df.at[row_index, col_name] = current_file_path

        # updating current row index
        current_row_index += 1


def get_autophagy_level(foci_count: int,
                        foci_area_mean: float,
                        foci_count_threshold: int,
                        foci_area_mean_threshold: float
                        ) -> str:
    """
    Given parameters for determining
    autophagy levels, calculates it
    and returns autophagy level string.
    """
    # defining placeholder value for autophagy level
    autophagy_level = 'undefined'

    # TODO: remove once test completed
    autophagy_level = 'LowAutophagy' if foci_count == 0 else 'HighAutophagy'
    return autophagy_level

    # getting conditional bools
    no_foci = foci_count == 0
    many = foci_count >= foci_count_threshold
    big = foci_area_mean >= foci_area_mean_threshold
    many_small = many and (not big)
    many_big = many and big
    few_small = (not many) and (not big)
    few_big = (not many) and big

    # checking conditionals

    if no_foci:

        # updating autophagy level string
        autophagy_level = 'no_foci'

    elif many_small:

        # updating autophagy level string
        autophagy_level = 'many_small'

    elif many_big:

        # updating autophagy level string
        autophagy_level = 'many_big'

    elif few_small:

        # updating autophagy level string
        autophagy_level = 'few_small'

    elif few_big:

        # updating autophagy level string
        autophagy_level = 'few_big'

    # returning autophagy level
    return autophagy_level


def get_dna_damage_level(foci_count: int,
                         foci_area_mean: float,
                         foci_count_threshold: int,
                         foci_area_mean_threshold: float
                         ) -> str:
    """
    Given parameters for determining
    dna damage levels, calculates it
    and returns dna damage level string.
    """
    # defining placeholder value for dna damage level
    dna_damage_level = 'undefined'

    # getting conditional bools
    mitosis_incompatible = foci_count >= 3

    # updating dna damage level string
    dna_damage_level = 'HighDamage' if mitosis_incompatible else 'LowDamage'

    # returning dna damage level
    return dna_damage_level


def get_pixels_in_nucleus(image: ndarray,
                          width: float,
                          height: float
                          ) -> Series:
    """
    Given an image, and coordinates
    for nucleus, returns pixels values
    inside it.
    """
    # getting image shape/dimensions
    image_shape = image.shape
    image_width = image_shape[1]
    image_height = image_shape[0]

    # getting cx/cy values
    cx = image_width / 2
    cy = image_height / 2
    cx = int(cx)
    cy = int(cy)

    # creating blank image (which will hold contour mask)
    blank_image = get_blank_image(width=image_width,
                                  height=image_height)

    # drawing contour in blank image
    draw_ellipse(open_img=blank_image,
                 cx=cx,
                 cy=cy,
                 width=height,
                 height=width,
                 angle=0.0,
                 color=1,
                 thickness=-1)

    # getting pixel coordinates matching contour color
    contour_pixel_coords = np_where(blank_image == 1)

    # getting respective pixels in base image
    pixels_in_contour = image[contour_pixel_coords[0], contour_pixel_coords[1]]

    # converting pixels to Series object
    pixels_in_contour = Series(pixels_in_contour)

    # returning respective pixels
    return pixels_in_contour


def get_pixels_in_ring(image: ndarray,
                       nucleus_width: float,
                       nucleus_height: float,
                       expanded_width: float,
                       expanded_height: float
                       ) -> ndarray:
    """
    Given an image, and coordinates
    for nucleus/cytoplasm ring, returns
    pixels inside ring.
    """
    # getting image shape/dimensions
    image_shape = image.shape
    image_width = image_shape[1]
    image_height = image_shape[0]

    # getting cx/cy values
    cx = image_width / 2
    cy = image_height / 2
    cx = int(cx)
    cy = int(cy)

    # creating blank image (which will hold contour mask)
    blank_image = get_blank_image(width=image_width,
                                  height=image_height)

    # drawing expanded nucleus contour in blank image
    draw_ellipse(open_img=blank_image,
                 cx=cx,
                 cy=cy,
                 width=expanded_height,
                 height=expanded_width,
                 angle=0.0,
                 color=1,
                 thickness=-1)

    # drawing nucleus contour in blank image (removing nucleus)
    draw_ellipse(open_img=blank_image,
                 cx=cx,
                 cy=cy,
                 width=nucleus_height,
                 height=nucleus_width,
                 angle=0.0,
                 color=0,
                 thickness=-1)

    # getting pixel coordinates matching contour color
    contour_pixel_coords = np_where(blank_image == 1)

    # getting respective pixels in base image
    pixels_in_contour = image[contour_pixel_coords[0], contour_pixel_coords[1]]

    # returning respective pixels
    return pixels_in_contour


def get_erk_ratio(crop_path: str,
                  width: float,
                  height: float,
                  ring_expansion: float
                  ) -> float:
    """
    Given a path to a crop, and
    coordinates for respective nucleus,
    returns nucleus/cytoplasm ratio
    """
    # reading image as grayscale
    image = load_grayscale_img(image_path=crop_path)

    # getting nucleus pixels
    nucleus_pixels = get_pixels_in_nucleus(image=image,
                                           width=width,
                                           height=height)

    # getting ring expansion dimensions
    expanded_width = width * ring_expansion
    expanded_height = height * ring_expansion

    # getting ring pixels
    nucleus_ring_pixels = get_pixels_in_ring(image=image,
                                             nucleus_width=width,
                                             nucleus_height=height,
                                             expanded_width=expanded_width,
                                             expanded_height=expanded_height)

    # getting pixels mean intensities
    nucleus_mean = nucleus_pixels.mean()
    ring_mean = nucleus_ring_pixels.mean()

    # getting current crop erk ratio
    erk_ratio = ring_mean / nucleus_mean

    # returning crop ratio
    return erk_ratio


def get_erk_level(nucleus_cytoplasm_ratio: float) -> str:
    """
    Given parameters for determining
    erk levels, calculates it and
    returns erk level string.
    """
    # defining placeholder value for erk level
    erk_level = 'undefined'

    # checking nucleus/cytoplasm ratio
    if nucleus_cytoplasm_ratio > 1.0:  # nucleus is brighter

        # updating erk level string
        erk_level = 'Inactive'

    else:  # cytoplasm is brighter

        # updating erk level string
        erk_level = 'Active'

    # returning erk level
    return erk_level


def get_cell_cycle_binary(red_value: float,
                          green_value: float
                          ) -> str:
    """
    Given red and green channels' pixel values,
    returns cell cycle, based on 0-1 scale
    for pixels intensity and given min values.
    """
    # defining placeholder value for cell cycle
    cell_cycle = None

    if red_value >= green_value:

        # then, cell cycle must be 'G1' (red)
        cell_cycle = 'G1'

    else:

        # then, cell cycle must be 'G2' (green)
        cell_cycle = 'G2'

    # returning cell cycle
    return cell_cycle


def get_cell_cycle(red_value: float,
                   green_value: float,
                   min_red_value: float,
                   min_green_value: float,
                   ratio_lower_threshold: float,
                   ratio_upper_threshold: float
                   ) -> str:
    """
    Given red and green channels' pixel values,
    returns cell cycle, based on 0-1 scale
    for pixels intensity and given min values.
    """
    # defining placeholder value for cell cycle
    cell_cycle = None

    # getting min values bool
    just_red = (red_value >= min_red_value) and (green_value < min_green_value)
    just_green = (red_value < min_red_value) and (green_value >= min_green_value)
    neither_reach_min = (red_value < min_red_value) and (green_value < min_green_value)
    both_reach_min = (red_value >= min_red_value) and (green_value >= min_green_value)

    # checking whether pixels reached min level

    if neither_reach_min:

        # then, cell cycle must be 'M-eG1'
        cell_cycle = 'M-eG1'

    elif just_red:

        # then, cell cycle must be 'G1' (red)
        cell_cycle = 'G1'

    elif just_green:

        # then, cell cycle must be 'G2' (green)
        cell_cycle = 'G2'

    elif both_reach_min:

        # calculating pixels' intensity ratio
        pixel_ratio = red_value / green_value

        # checking ratio
        if pixel_ratio > ratio_upper_threshold:

            # then, cell cycle must be 'G1' (red)
            cell_cycle = 'G1'

        elif pixel_ratio < ratio_lower_threshold:

            # then, cell cycle must be 'G2' (green)
            cell_cycle = 'G2'

        else:

            # then, cell cycle must be 'S' (red AND green)
            cell_cycle = 'S'

    # returning cell cycle
    return cell_cycle


def add_cell_cycle_col(df: DataFrame,
                       min_red_value: float,
                       min_green_value: float,
                       ratio_lower_threshold: float,
                       ratio_upper_threshold: float
                       ):
    """
    Given an analysis data frame, and
    min levels for red/green pixel values,
    adds cell cycle col, based on pixel
    intensities ratio.
    """
    # defining col name
    col_name = 'cell_cycle'

    # adding placeholder values to col
    df[col_name] = None

    # getting df rows
    df_rows = df.iterrows()

    # getting rows num
    rows_num = len(df)

    # defining starter for current row index
    current_row_index = 1

    # iterating over rows
    for row in df_rows:

        # printing progress message
        base_string = f'adding cell cycle col (row #INDEX# of #TOTAL#)'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row index/data
        row_index, row_data = row

        # getting current row red/green values
        red_value = row_data['Mean_red']
        green_value = row_data['Mean_green']

        # getting current row cell cycle
        # current_cell_cycle = get_cell_cycle(red_value=red_value,
        #                                     green_value=green_value,
        #                                     min_red_value=min_red_value,
        #                                     min_green_value=min_green_value,
        #                                     ratio_lower_threshold=ratio_lower_threshold,
        #                                     ratio_upper_threshold=ratio_upper_threshold)
        current_cell_cycle = get_cell_cycle_binary(red_value=red_value,
                                                   green_value=green_value)

        # updating current row col
        df.at[row_index, col_name] = current_cell_cycle

        # updating current row index
        current_row_index += 1


def add_cell_cycle_proportions_col(df: DataFrame) -> None:
    """
    Given an analysis data frame,
    adds cell cycle proportions col.
    """
    # defining col name
    col_name = 'cell_cycle (%cells)'

    # adding placeholder values to col
    df[col_name] = None

    # defining group cols
    group_cols = ['treatment', 'date', 'time']

    # grouping df by treatment
    df_groups = df.groupby(group_cols)

    # getting rows num
    rows_num = len(df)

    # defining placeholder value for current_row_index
    current_row_index = 1

    # iterating over df groups
    for df_name, df_group in df_groups:

        # getting values from df name
        current_treatment, current_date, current_time = df_name

        # getting current treatment total cells count
        total_cells_count = len(df_group)

        # grouping df by cell cycle
        cell_cycle_groups = df_group.groupby('cell_cycle')

        # iterating over cell cycle group
        for cell_cycle, cell_cycle_group in cell_cycle_groups:

            # getting current cell cycle cell count
            current_cells_count = len(cell_cycle_group)

            # getting current group proportion
            current_group_ratio = current_cells_count / total_cells_count
            current_group_percentage = current_group_ratio * 100
            current_group_percentage_round = round(current_group_percentage)
            current_group_percentage_str = f'{cell_cycle} ({current_group_percentage_round}%)'

            # getting current group rows
            df_rows = cell_cycle_group.iterrows()

            # iterating over current group rows
            for row_index, row_data in df_rows:

                # printing progress message
                base_string = f'adding cell cycle proportions col (row #INDEX# of #TOTAL#)'
                print_progress_message(base_string=base_string,
                                       index=current_row_index,
                                       total=rows_num)

                # updating current group cell cycle proportion col
                df.at[row_index, col_name] = current_group_percentage_str

                # updating row index
                current_row_index += 1


def get_pixel_intensity(file_path: str,
                        calc: str
                        ) -> float:
    """
    Given a file path, loads image
    and returns pixel intensity value,
    based on given calc method (mean, min, max).
    """
    # loading image
    img = load_grayscale_img(image_path=file_path)

    # defining placeholder value for current intensity value
    pixel_intensity = None

    # getting current intensity value based on given calc str

    # calculating min intensity
    if calc == 'min':
        pixel_intensity = img.min()

    # calculating max intensity
    elif calc == 'max':
        pixel_intensity = img.max()

    # calculating mean intensity
    elif calc == 'mean':
        pixel_intensity = img.mean()

    # calculating intensities ratio ('het')
    elif calc == 'het_mean':
        pixel_max = img.max()
        pixel_mean = img.mean()
        pixel_intensity = pixel_max / pixel_mean

    # calculating intensities ratio ('het')
    elif calc == 'het_median':
        pixel_max = img.max()
        pixel_median = img.mean()
        pixel_intensity = pixel_max / pixel_median

    else:

        # printing execution message
        f_string = f'calc mode {calc} not specified.\n'
        f_string += f'Please, check and try again.'
        print(f_string)

        # quitting
        exit()

    # converting pixel intensity to float
    pixel_intensity = float(pixel_intensity)

    # returning pixel intensity value
    return pixel_intensity


def get_nucleus_pixel_intensity(crop_path: str,
                                nucleus_width: float,
                                nucleus_height: float,
                                calc: str
                                ) -> float:
    """
    Given a path to a crop, loads image
    and returns pixel intensity value,
    based on given calc method (mean, min, max).
    """
    # defining placeholder value for pixel intensity
    pixel_intensity = None

    # loading image
    img = load_grayscale_img(image_path=crop_path)

    # getting pixels in nucleus
    nucleus_pixels = get_pixels_in_nucleus(image=img,
                                           width=nucleus_width,
                                           height=nucleus_height)

    # checking calc mode
    if calc == 'mean':

        # updating pixel intensity
        pixel_intensity = nucleus_pixels.mean()

    # checking calc mode
    elif calc == 'median':

        # updating pixel intensity
        pixel_intensity = nucleus_pixels.median()

    # checking calc mode
    elif calc == 'sum':

        # updating pixel intensity
        pixel_intensity = nucleus_pixels.sum()

    else:

        # printing execution message
        f_string = f'calc mode {calc} not specified.\n'
        f_string += f'Please, check and try again.'
        print(f_string)

        # quitting
        exit()

    # converting pixel intensity to float
    pixel_intensity = float(pixel_intensity)

    # returning pixel intensity value
    return pixel_intensity


def augment_image(image_name: str,
                  extension: str,
                  images_folder: str,
                  output_folder: str,
                  resize: bool
                  ) -> None:
    # adding extension to image name
    image_name = f'{image_name}{extension}'

    # defining base save path
    save_path = join(output_folder,
                     image_name)

    # getting final save path (used to check current image augmentation)
    final_save_path = save_path.replace(extension, f'_hu{extension}')

    # checking whether current save path already exists
    if exists(final_save_path):

        # skipping current image
        return None

    # getting current image path
    image_path = join(images_folder,
                      image_name)

    # opening current image
    open_image = imread(image_path)

    # checking resize toggle
    if resize:

        # resizing image
        open_image = resize_image(open_image=open_image,
                                  image_size=IMAGE_SIZE)

    # getting rotated image
    rotated_image = rotate(open_image,
                           ROTATE_180)

    # getting vertically flipped image
    v_flipped_image = flip(open_image,
                           0)

    # getting horizontally flipped image
    h_flipped_image = flip(open_image,
                           1)

    # defining alpha and beta
    alpha_d = 0.9  # Contrast control
    beta_d = -2    # Brightness control
    alpha_u = 1.1  # Contrast control
    beta_u = 5     # Brightness control

    # getting contrast/brightness changed image
    od_contrast_image = convertScaleAbs(open_image,
                                        alpha=alpha_d,
                                        beta=beta_d)

    rd_contrast_image = convertScaleAbs(rotated_image,
                                        alpha=alpha_d,
                                        beta=beta_d)

    vd_contrast_image = convertScaleAbs(v_flipped_image,
                                        alpha=alpha_d,
                                        beta=beta_d)

    hd_contrast_image = convertScaleAbs(h_flipped_image,
                                        alpha=alpha_d,
                                        beta=beta_d)

    ou_contrast_image = convertScaleAbs(open_image,
                                        alpha=alpha_u,
                                        beta=beta_u)

    ru_contrast_image = convertScaleAbs(rotated_image,
                                        alpha=alpha_u,
                                        beta=beta_u)

    vu_contrast_image = convertScaleAbs(v_flipped_image,
                                        alpha=alpha_u,
                                        beta=beta_u)

    hu_contrast_image = convertScaleAbs(h_flipped_image,
                                        alpha=alpha_u,
                                        beta=beta_u)

    # saving images
    imwrite(save_path.replace(extension, f'_o{extension}'), open_image)
    imwrite(save_path.replace(extension, f'_r{extension}'), rotated_image)
    imwrite(save_path.replace(extension, f'_v{extension}'), v_flipped_image)
    imwrite(save_path.replace(extension, f'_h{extension}'), h_flipped_image)
    imwrite(save_path.replace(extension, f'_od{extension}'), od_contrast_image)
    imwrite(save_path.replace(extension, f'_rd{extension}'), rd_contrast_image)
    imwrite(save_path.replace(extension, f'_vd{extension}'), vd_contrast_image)
    imwrite(save_path.replace(extension, f'_hd{extension}'), hd_contrast_image)
    imwrite(save_path.replace(extension, f'_ou{extension}'), ou_contrast_image)
    imwrite(save_path.replace(extension, f'_ru{extension}'), ru_contrast_image)
    imwrite(save_path.replace(extension, f'_vu{extension}'), vu_contrast_image)
    imwrite(save_path.replace(extension, f'_hu{extension}'), hu_contrast_image)


def train_model(model,
                train_data,
                val_data,
                epochs,
                callback
                ):
    """
    Given a model, data, and execution
    parameters, trains model on given
    data, returning train history.
    """
    # training model (and storing history)
    print('training model...')
    train_history = model.fit(train_data,
                              epochs=epochs,
                              validation_data=val_data,
                              callbacks=[callback])
    print('training complete!')

    # getting train history dict
    history_dict = train_history.history

    # returning train history
    return history_dict


def get_history_df(history_dict):
    """
    Given a history dict (output from
    train function), returns a history
    df (useful for plotting data).
    """
    # creating data frame based on dict
    history_df = DataFrame(history_dict)

    # getting epochs num
    epochs_num = len(history_df)

    # getting epochs range
    epochs_range = range(epochs_num)

    # getting epochs column
    epochs_col = [f + 1 for f in epochs_range]

    # adding epoch column
    history_df['epoch'] = epochs_col

    # melting df
    history_df = history_df.melt('epoch')

    # returning df
    return history_df


def generate_history_plot(df: DataFrame,
                          logs_folder: str,
                          epochs_num: int
                          ) -> None:
    """
    Given a train history df,
    generates history plot,
    saving it to given logs folder.
    """
    # getting save path
    save_name = 'train_history.png'
    save_path = join(logs_folder,
                     save_name)

    # plotting history
    print('plotting history...')
    lineplot(data=df,
             x='epoch',
             y='value',
             hue='variable')

    # defining axis lims
    y_min, y_max = (0.0, 1.0)
    x_min, x_max = (0, epochs_num)

    # setting axis lims
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # setting x ticks
    epochs_range = range(epochs_num)
    x_ticks = [e for e in epochs_range]
    plt.xticks(x_ticks)

    # setting plot title
    title = 'Train History'
    plt.title(title)

    # adjusting plot layout
    plt.tight_layout()

    # saving figure
    fig_path = join(save_path)
    plt.savefig(fig_path)


def load_bgr_img(image_path: str) -> ndarray:
    """
    Given a path to an image,
    returns image as BGR ndarray.
    """
    # opening current image
    current_image = imread(image_path,
                           -1)

    # checking current image shape (grayscale/RGB)
    image_shape = current_image.shape
    image_len = len(image_shape)

    # checking image type
    if image_len < 3:  # not rgb (grayscale)

        # converting current image to rgb
        current_image = cvtColor(current_image, COLOR_GRAY2BGR)

    # returning image
    return current_image


def load_grayscale_img(image_path: str) -> ndarray:
    """
    Given a path to an image,
    returns image as grayscale ndarray.
    """
    # opening current image
    current_image = imread(image_path,
                           IMREAD_GRAYSCALE)

    # returning image
    return current_image


def get_crops_df(crops_file: str) -> DataFrame:
    """
    Given a path to a crops info csv,
    returns crops data frame.
    :param crops_file: String. Represents a path to a file.
    :return: DataFrame. Represents crops info data frame.
    """
    # defining col types
    col_types = {'img_name': str,
                 'crop_index': int,
                 'crop_name': str,
                 'cx': int,
                 'cy': int,
                 'width': int,
                 'height': int,
                 'angle': float,
                 'class': str}

    # reading crops file
    crops_df = read_csv(crops_file,
                        dtype=col_types)

    # returning crops df
    return crops_df


def get_contour_centroid(contour: ndarray) -> tuple:
    """
    Given a contour, returns
    center coordinates in a tuple
    of following structure:
    (cx, cy)
    """
    # getting contour coords
    contour_coords = boundingRect(contour)

    # extracting features from coords tuple
    corner_x, corner_y, width, height = contour_coords

    # getting current center points
    cx = corner_x + (width / 2)
    cy = corner_y + (height / 2)

    # converting cx/cy to ints
    cx = int(cx)
    cy = int(cy)

    # assembling coords tuple
    coords_tuple = (cx, cy)

    # returning coords tuple
    return coords_tuple


def get_contours_df(image_name: str,
                    image_path: str,
                    contour_type: str
                    ) -> DataFrame:
    """
    Given a path to a binary image,
    finds contours and returns data
    frame containing contours coords.
    """
    # reading image
    image = imread(image_path,
                   -1)

    # finding contours in image
    contours, _ = findContours(image, RETR_EXTERNAL, CHAIN_APPROX_NONE)

    # getting number of contours found in image
    contours_num = len(contours)
    contours_num_range = range(contours_num)

    # getting current contours indices
    contours_indices = [f for f in contours_num_range]

    # getting current contours coords
    contours_coords = [get_contour_centroid(contour) for contour in contours]

    # getting current contours areas
    contours_areas = [contourArea(contour) for contour in contours]

    # getting current image col lists
    image_names = [image_name for _ in contours_num_range]
    contour_types = [contour_type for _ in contours_num_range]

    # assembling contours dict
    contours_dict = {'image_name': image_names,
                     'contour_index': contours_indices,
                     'contour': contours,
                     'contour_type': contour_types,
                     'coords': contours_coords,
                     'area': contours_areas}

    # assembling contours df
    contours_df = DataFrame(contours_dict)

    # returning contours df
    return contours_df


def show_image(image: ndarray,
               window_name: str = 'test_window'
               ) -> None:
    """
    Given a numpy array, show it
    in screen, using given window
    name as title.
    """
    # showing image
    imshow(window_name, image)

    # waiting fow user input (required for cv2.imshow to work)
    waitKey(0)

    # closing windows
    destroyAllWindows()


def get_prediction(prediction: float,
                   phenotype: str
                   ) -> float or str:
    """
    Given a prediction value (float/ndarray),
    and a phenotype, returns respective phenotype
    string (or float in case of nii).
    """
    # defining placeholder value for current prediction
    current_prediction = None

    # getting current prediction value based on given phenotype
    # always follows alphanumeric, just as train_data.class_indices
    # in training module.

    if phenotype == 'nii':

        # converting prediction to float
        current_prediction = float(prediction)

        # adjusting scale (based on training data normalizer)
        current_prediction = current_prediction * 20

    elif phenotype == 'cell_cycle':

        # converting prediction to string
        current_prediction = 'G1' if prediction < 0.5 else 'G2'

    elif phenotype == 'autophagy':

        # converting prediction to string
        current_prediction = 'HighAutophagy' if prediction < 0.5 else 'LowAutophagy'

    elif phenotype == 'erk':

        # converting prediction to string
        current_prediction = 'Active' if prediction < 0.5 else 'Inactive'

    elif phenotype == 'dna_damage':

        # converting prediction to string
        current_prediction = 'HighDamage' if prediction < 0.5 else 'LowDamage'

    elif phenotype == 'cats_dogs':

        # converting prediction to string
        current_prediction = 'Cat' if prediction < 0.5 else 'Dog'

    else:

        # printing execution message
        f_string = f'phenotype {phenotype} not specified.\n'
        f_string += f'Please, check and try again.'
        print(f_string)

        # quitting
        exit()

    # returning prediction value
    return current_prediction


def get_pearson_correlation(df: DataFrame,
                            real_col: str,
                            pred_col: str
                            ) -> float:
    """
    Given an errors data frame,
    returns pearson correlation.
    """
    # getting real/prediction cols
    real_col = df[real_col]
    pred_col = df[pred_col]

    # calculating pearson correlation
    print('calculating pearson correlation...')
    corr_value, p_value = pearsonr(x=real_col,
                                   y=pred_col)

    # assembling results string
    f_string = '--Pearson correlation test results--\n'
    f_string += f'p-value: {p_value}\n'
    f_string += f's-value: {corr_value}'

    # printing test results
    spacer()
    print(f_string)
    spacer()

    # returning correlation value
    return corr_value


def run_levene_test(samples: list) -> None:
    """
    Given a list of samples,
    runs Levene test.
    """
    # calculating Levene statistic
    print('calculating Levene statistic...')
    s_value, p_value = levene(*samples)

    # assembling results string
    f_string = '--Levene test results--\n'
    f_string += f'p-value: {p_value}\n'
    f_string += f's-value: {s_value}'

    # printing test results
    spacer()
    print(f_string)
    spacer()


def run_anova_test(samples: list) -> None:
    """
    Given a list of samples,
    runs one-way ANOVA test.
    """
    # calculating ANOVA statistic
    print('calculating one-way ANOVA statistic...')
    s_value, p_value = f_oneway(*samples)

    # assembling results string
    f_string = '--ANOVA test results--\n'
    f_string += f'p-value: {p_value}\n'
    f_string += f's-value: {s_value}'

    # printing test results
    spacer()
    print(f_string)
    spacer()


def run_kruskal_test(samples: list) -> None:
    """
    Given a list of samples,
    runs Kruskal test.
    """
    # calculating Kruskal statistic
    print('calculating Kruskal statistic...')
    s_value, p_value = kruskal(*samples)

    # assembling results string
    f_string = '--Kruskal test results--\n'
    f_string += f'p-value: {p_value}\n'
    f_string += f's-value: {s_value}'

    # printing test results
    print(f_string)
    spacer()


def run_mannwhitneyu_test(sample_a: list,
                          sample_b: list
                          ) -> None:
    """
    Given a list of samples,
    runs mann-whitney-U test.
    """
    # getting sample info
    sample_a_len = len(sample_a)
    sample_b_len = len(sample_b)

    # calculating Mann-Whitney-U statistic
    print('calculating Mann-Whitney-U statistic...')
    s_value, p_value = mannwhitneyu(x=sample_a,
                                    y=sample_b)

    # assembling results string
    f_string = '--Mann-Whitney-U test results--\n'
    f_string += f'sample_a count: {sample_a_len}\n'
    f_string += f'sample_b count: {sample_b_len}\n'
    f_string += f'p-value: {p_value}\n'
    f_string += f's-value: {s_value}'

    # printing test results
    print(f_string)
    spacer()


def run_paired_test(sample_a: list,
                    sample_b: list
                    ) -> None:
    """
    Given a list of samples,
    runs paired t-test.
    """
    # calculating paired t-test statistic
    print('calculating paired t-test statistic...')
    s_value, p_value = ttest_rel(a=sample_a,
                                 b=sample_b)

    # assembling results string
    f_string = '--paired t-test test results--\n'
    f_string += f'p-value: {p_value}\n'
    f_string += f's-value: {s_value}'

    # printing test results
    print(f_string)
    spacer()

######################################################################
# end of current module
