# generate table from crops module

print('initializing...')  # noqa

# Code destined to generating pixel intensity
# histograms - Fucci related.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import get_crop_pixels
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import drop_unrequired_cols
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'generate pixel intensity histograms for single-cell crops'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # red folder param
    parser.add_argument('-r', '--red-folder',
                        dest='red_folder',
                        required=True,
                        help='defines red input folder (folder containing crops in fluorescence channel)')

    # green folder param
    parser.add_argument('-g', '--green-folder',
                        dest='green_folder',
                        required=True,
                        help='defines green input folder (folder containing crops in fluorescence channel)')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

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


def generate_pixel_intensity_histograms(red_folder: str,
                                        green_folder: str,
                                        images_extension: str,
                                        crops_file: str,
                                        output_folder: str,
                                        ) -> None:
    """
    Given a path to a folder containing crops,
    and a path to a file containing crops info,
    generates ML input compatible tables
    :param red_folder: String. Represents a path to a folder.
    :param green_folder: String. Represents a path to a folder.
    :param images_extension: String. Represents image extension.
    :param crops_file: String. Represents a path to a file.
    :param output_folder: String. Represents a path to a folder.
    :return: None.
    """
    # getting crops df
    crops_df = get_crops_df(crops_file=crops_file)

    print(crops_df)
    exit()

    # getting final df
    final_df = DataFrame()

    # saving final df
    save_name = f'crops_ml_df.pickle'
    save_path = join(output_folder,
                     save_name)
    final_df.to_csv(save_path)

    # printing execution message
    print(f'files saved to {output_folder}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting red folder
    red_folder = args_dict['red_folder']

    # getting green folder
    green_folder = args_dict['green_folder']

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

    # running generate_pixel_intensity_histograms function
    generate_pixel_intensity_histograms(red_folder=red_folder,
                                        green_folder=green_folder,
                                        images_extension=images_extension,
                                        crops_file=crops_file,
                                        output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
