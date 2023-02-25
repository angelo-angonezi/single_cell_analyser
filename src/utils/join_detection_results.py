# join detection results module

# given a path to a folder containing model detection files
# (multiple det_*class_name*.txt) joins them into a single file.

######################################################################
# imports

from os import listdir
from os.path import join
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from os.path import split as os_split
from src.utils.aux_funcs import flush_or_print

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "join detection results"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        type=str,
                        help='defines input folder (folder containing detection files [normal and round txts])')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        type=str,
                        help='defines output path (path to .csv file)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_detection_files_paths(folder_path: str) -> list:
    """
    Given a path to a folder containing detection files,
    returns sorted paths for detection files (csvs).
    :param folder_path: String. Represents a path to a folder.
    :return: List. Represents detection files' paths.
    """
    # getting detection files in folder (only csv files)
    detection_files = [join(folder_path, file)    # getting file path
                       for file                   # iterating over files
                       in listdir(folder_path)    # in input directory
                       if file.endswith('.txt')]  # if file matches extension ".txt"

    # sorting paths
    sorted(detection_files)

    # returning files paths
    return detection_files


def create_dataframe_from_multiple_detection_files(input_folder: str,
                                                   output_path: str
                                                   ) -> None:
    """
    Given a path to a folder containing multiple detection files,
    saves an ordered DataFrame in which each row corresponds to a
    single detection, adding a new column corresponding
    to detection class.
    :param input_folder: String. Represents a path to a folder.
    :param output_path: String. Represents a path to an output file.
    :return: DataFrame. Represents multiple cell detections info.
    """
    # printing execution message
    print('creating main dataframe...')

    # defining placeholder value for dfs_list
    dfs_list = []

    # defining column names
    column_names = []

    # getting detection files in input folder
    detection_files = get_detection_files_paths(folder_path=input_folder)

    # printing execution message
    print('adding detections to dataframe...')

    # iterating over detection files
    for detection_file in detection_files:

        # checking file class
        file_name_split = detection_file.split('_')
        det_class = file_name_split[1]

        # opening file
        detection_file_df = read_csv(detection_file,
                                     sep=' ',
                                     header=None)

        # getting rows in input df
        rows = detection_file_df.iterrows()

        # getting number of rows
        rows_num = len(detection_file_df)

        # iterating over rows in input df
        for row in rows:

            # getting row index/data
            row_index, row_data = row

            # correcting index
            row_index += 1

            # flushing execution message
            f_string = f'reading row {row_index} of {rows_num} ({det_class})...'
            flush_or_print(string=f_string,
                           index=row_index,
                           total=rows_num)

            # getting current row info
            img_file_name = row_data[0]
            detection_threshold = round(row_data[1], 3)
            cx = row_data[2]
            cy = row_data[3]
            width = row_data[4]
            height = row_data[5]
            angle = row_data[6]

            # creating current row dictionary
            current_row_dict = {'img_file_name': img_file_name,
                                'detection_threshold': detection_threshold,
                                'cx': cx,
                                'cy': cy,
                                'width': width,
                                'height': height,
                                'angle': angle,
                                'class': det_class}

            # converting current row dict to df
            current_row_df = DataFrame(current_row_dict,
                                       index=[0])

            # appending current df to dfs_list
            dfs_list.append(current_row_df)

    # concatenating dfs in dfs_list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # sorting final dataframe
    sorted_df = final_df.sort_values(['img_file_name'])

    # printing execution message
    print('saving dataframe...')

    # saving dataframe to output path
    sorted_df.to_csv(output_path,
                     index=False)

    # printing execution message
    print(f'dataframe saved at "{output_path}"')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input folder
    input_folder = args_dict['input_folder']
    print(input_folder)

    # getting output folder
    output_name = os_split(input_folder)[-1]
    print(output_name)
    output_name = f'{output_name}_detections.csv'
    output_path = join(input_folder, output_name)

    # running create dataframe function
    create_dataframe_from_multiple_detection_files(input_folder=input_folder,
                                                   output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
