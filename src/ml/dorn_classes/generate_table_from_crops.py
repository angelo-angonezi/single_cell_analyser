# generate table from crops module

print('initializing...')  # noqa

# Code destined to generating ML input
# table, based on crops info.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from pandas import Series
from pandas import read_csv
from pandas import DataFrame
from numpy import sort as np_sort
from argparse import ArgumentParser
from src.utils.aux_funcs import get_crop_pixels
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import drop_unrequired_cols
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'generate ML table from crops module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    input_help = 'defines input folder (folder containing crops)'
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help=input_help)

    # images_extension param
    images_extension_help = 'defines extension (.tif, .png, .jpg) of images in input folders'
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help=images_extension_help)

    # crops file param
    crops_help = 'defines path to crops file (containing crops info)'
    parser.add_argument('-c', '--crops-file',
                        dest='crops_file',
                        required=True,
                        help=crops_help)

    # output folder param
    output_help = 'defines output folder (folder that will contain output files)'
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help=output_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


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

    # defining cols to keep
    cols_to_keep = ['crop_name', 'class']

    # dropping unrequired cols
    drop_unrequired_cols(df=crops_df,
                         cols_to_keep=cols_to_keep)

    # returning crops df
    return crops_df


def get_crop_path(row_data: Series,
                  input_folder: str,
                  images_extension: str
                  ) -> str:
    """
    Given a crops data frame row, and
    a path to input folder+images extension,
    returns given crop path.
    :param row_data: Series. Represents a crops data frame row.
    :param input_folder: String. Represents a path to a folder.
    :param images_extension: String. Represents image extension.
    :return: String. Represents a path to a crop.
    """
    # getting current row crop name
    crop_name = row_data['crop_name']

    # getting current crop name+extension
    crop_name_w_extension = f'{crop_name}{images_extension}'

    # getting current row crop path
    crop_path = join(input_folder,
                     crop_name_w_extension)

    # returning crop path
    return crop_path


def get_crop_class(row_data: Series) -> str:
    """
    Given a crops data frame row,
    returns given crop class.
    :param row_data: Series. Represents a crops data frame row.
    :return: String. Represents a crop's class.
    """
    # getting current row crop name
    crop_class = row_data['class']

    # returning crop class
    return crop_class


def get_crops_ml_df(input_folder: str,
                    images_extension: str,
                    crops_df: DataFrame
                    ) -> DataFrame:
    """
    Given a crops data frame, returns
    ml input ready data frame.
    :param input_folder: String. Represents a path to a folder.
    :param images_extension: String. Represents image extension.
    :param crops_df: DataFrame. Represents a crops data frame.
    :return: DataFrame. Represents a ml crops data frame.
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # getting rows num
    rows_num = len(crops_df)

    # getting df rows
    df_rows = crops_df.iterrows()

    # defining progress base string
    progress_string = 'converting crop #INDEX# of #TOTAL#'

    # iterating over df rows (crops)
    for row_index, row_data in df_rows:

        # converting row index to int
        row_index = int(row_index)

        # getting adapted current row index (required for print_progress_message)
        row_index += 1

        # printing execution message
        print_progress_message(base_string=progress_string,
                               index=row_index,
                               total=rows_num)

        # getting current crop path
        current_crop_path = get_crop_path(row_data=row_data,
                                          input_folder=input_folder,
                                          images_extension=images_extension)

        # getting current crop class
        current_crop_class = get_crop_class(row_data=row_data)

        # getting current crop pixels
        current_crop_pixels = get_crop_pixels(crop_path=current_crop_path)

        # getting current crop sorted pixels
        current_crop_sorted_pixels = np_sort(current_crop_pixels)

        # assembling current cell dict
        current_dict = {'crop_path': current_crop_path,
                        'class': current_crop_class,
                        'pixels': [current_crop_pixels],
                        'sorted_pixels': [current_crop_sorted_pixels]}

        # assembling current cell df
        current_df = DataFrame(current_dict)

        # appending current df to dfs list
        dfs_list.append(current_df)

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def generate_table_from_crops(input_folder: str,
                              images_extension: str,
                              crops_file: str,
                              output_folder: str,
                              ) -> None:
    """
    Given a path to a folder containing crops,
    and a path to a file containing crops info,
    generates ML input compatible tables
    :param input_folder: String. Represents a path to a folder.
    :param images_extension: String. Represents image extension.
    :param crops_file: String. Represents a path to a file.
    :param output_folder: String. Represents a path to a folder.
    :return: None.
    """
    # getting crops df
    crops_df = get_crops_df(crops_file=crops_file)

    # getting crops ml df
    crops_ml_df = get_crops_ml_df(input_folder=input_folder,
                                  images_extension=images_extension,
                                  crops_df=crops_df)

    # saving final df
    save_name = f'crops_ml_df.pickle'
    save_path = join(output_folder,
                     save_name)
    crops_ml_df.to_pickle(save_path)

    # printing execution message
    print(f'files saved to {output_folder}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input folder
    input_folder = args_dict['input_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting crops file
    crops_file = args_dict['crops_file']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_table_from_crops function
    generate_table_from_crops(input_folder=input_folder,
                              images_extension=images_extension,
                              crops_file=crops_file,
                              output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
