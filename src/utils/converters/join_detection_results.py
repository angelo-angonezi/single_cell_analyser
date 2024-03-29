# join detection results module

print('initializing...')  # noqa

# Given a path to a folder containing model detection files
# (multiple det_*class_name*.txt) joins them into a single file.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os import environ
from os.path import join
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from pandas.errors import EmptyDataError
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

# setting tensorflow warnings off
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


def create_dataframe_from_multiple_detection_files(input_folder: str) -> DataFrame:
    """
    Given a path to a folder containing multiple detection files,
    saves an ordered DataFrame in which each row corresponds to a
    single detection, adding a new column corresponding
    to detection class.
    :param input_folder: String. Represents a path to a folder.
    :return: DataFrame. Represents multiple cell detections info.
    """
    # defining placeholder value for dfs_list
    dfs_list = []

    # getting detection files in input folder
    print('getting detection files in input folder...')
    detection_files = get_specific_files_in_folder(path_to_folder=input_folder,
                                                   extension='.txt')

    # iterating over detection files
    for detection_file in detection_files:

        # printing execution message
        f_string = f'getting detections from file "{detection_file}"...'
        print(f_string)

        # checking file class
        file_name_split = detection_file.split('_')
        det_class = file_name_split[1]

        # getting file path
        file_path = join(input_folder,
                         detection_file)

        # trying to open file
        try:

            # reading csv
            detection_file_df = read_csv(file_path,
                                         sep=' ',
                                         header=None)

        # if file is empty
        except EmptyDataError:

            # printing execution message
            f_string = f'No detections found in "{detection_file}" file. Skipping to next file...'
            print(f_string)

            # skipping current file
            continue

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
            f_string = f'reading row #INDEX# of #TOTAL#'
            print_progress_message(base_string=f_string,
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
    print('assembling joined dataframe...')
    final_df = concat(dfs_list,
                      ignore_index=True)

    # sorting final dataframe
    print('sorting joined dataframe...')
    sorted_df = final_df.sort_values(['img_file_name'])

    # returning sorted_df
    return sorted_df


def save_joined_dataframe(joined_df: DataFrame,
                          output_path: str
                          ) -> None:
    """
    Given a joined detections data frame, and a
    path to an output csv file, saves df in given
    output.
    :param joined_df: DataFrame. Represents joined detections df.
    :param output_path: String. Represents a path to an output file.
    :return: None.
    """
    # printing execution message
    print('saving dataframe...')

    # saving dataframe to output path
    joined_df.to_csv(output_path,
                     index=False)

    # printing execution message
    print(f'joined dataframe saved at "{output_path}"')

######################################################################
# defining main function


def main():
    """
    Gets arguments from cli and runs main code.
    """
    # getting args dict
    args_dict = get_args_dict()

    # getting input folder
    input_folder = args_dict['input_folder']

    # getting output path
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # creating joined data frame
    joined_df = create_dataframe_from_multiple_detection_files(input_folder=input_folder)

    # saving joined data frame
    save_joined_dataframe(joined_df=joined_df,
                          output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
