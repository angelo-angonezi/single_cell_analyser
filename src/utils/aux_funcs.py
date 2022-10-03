# auxiliary functions module

######################################################################
# importing required libraries
from sys import stdout
from os import listdir
from pandas import DataFrame

######################################################################
# defining auxiliary functions


def spacer(char: str = '_',
           reps: int = 50
           ) -> None:
    """
    Given a char and a number of reps,
    prints a "spacer" string assembled
    by multiplying char by reps.
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
    if index == total:  # current element is last

        # printing string
        print(string)

    else:  # current element is not last

        # flushing string
        flush_string(string)


def get_data_from_consolidated_df(consolidated_df_file_path: str) -> DataFrame:
    """
    Given a path to a consolidated dataframe,
    returns processed dataframe.
    """
    # printing execution message
    f_string = f'getting data from consolidated data frame csv...'
    print(f_string)

    # reading df from file path
    consolidated_df = read_csv(consolidated_df_file_path)

    # returning df
    return consolidated_df


def get_image_files_paths(folder_path: str) -> list:
    """
    Given a path to a folder containing image files,
    returns sorted paths for image files (.tif, .jpg, .png).
    :param folder_path: String. Represents a path to a folder.
    :return: List. Represents detection files' paths.
    """
    # printing execution message
    f_string = f'getting image paths...'
    spacer()
    print(f_string)

    # defining placeholder values for detection files list
    detection_files = []

    # getting tif files
    tif_files = [join(folder_path, file)    # getting file path
                 for file                   # iterating over files
                 in listdir(folder_path)    # in input directory
                 if file.endswith('.tif')]  # if file matches extension ".tif"

    # getting jpg files
    jpg_files = [join(folder_path, file)    # getting file path
                 for file                   # iterating over files
                 in listdir(folder_path)    # in input directory
                 if file.endswith('.jpg')]  # if file matches extension ".jpg"

    # getting png files
    png_files = [join(folder_path, file)    # getting file path
                 for file                   # iterating over files
                 in listdir(folder_path)    # in input directory
                 if file.endswith('.png')]  # if file matches extension ".png"

    # appending images to list
    detection_files.extend(tif_files)
    detection_files.extend(jpg_files)
    detection_files.extend(png_files)

    # sorting paths
    sorted(detection_files)

    # returning files paths
    return detection_files


def get_obbs_from_df(df: DataFrame) -> list:
    """
    Given a detections data frame, returns
    a list of detected centroids, in following
    format:
    [(cx, cy, width, height, angle), ...]
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


######################################################################
# end of current module
