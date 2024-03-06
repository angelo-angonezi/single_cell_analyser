# generate cell cycle annotations module

print('initializing...')  # noqa

# Code destined to generating cell
# cycle annotations file.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import get_cell_cycle
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_cell_cycle_simple
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_nucleus_pixel_intensity
print('all required libraries successfully imported.')  # noqa

######################################################################
# defining global variables

MIN_RED_VALUE = 0.15
MIN_GREEN_VALUE = 0.15
RATIO_LOWER_THRESHOLD = 0.8
RATIO_UPPER_THRESHOLD = 1.2

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'generate cell cycle annotations module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # crops file param
    parser.add_argument('-c', '--crops-file',
                        dest='crops_file',
                        required=True,
                        help='defines path to crops file (containing crops info)')

    # red folder param
    parser.add_argument('-r', '--red-folder',
                        dest='red_folder',
                        required=True,
                        help='defines red input folder (folder containing red crops)')

    # green folder param
    parser.add_argument('-g', '--green-folder',
                        dest='green_folder',
                        required=True,
                        help='defines green input folder (folder containing green crops)')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output file (.csv)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def add_cell_cycle_col(df: DataFrame,
                       red_folder: str,
                       green_folder: str,
                       images_extension: str
                       ) -> None:
    """
    Given a crops info df, and paths to
    red/green crops, adds cell cycle col
    based on red/green pixel intensities.
    """
    # defining col name
    col_name = 'class'

    # emptying class col
    df[col_name] = None

    # getting rows num
    rows_num = len(df)

    # getting df rows
    df_rows = df.iterrows()

    # defining starter for current row index
    current_row_index = 1

    # iterating over df rows
    for row_index, row_data in df_rows:

        # printing progress message
        base_string = 'adding cell cycle col to row #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row crop name
        crop_name = row_data['crop_name']

        # getting current row crop dimensions
        nucleus_width = row_data['width']
        nucleus_height = row_data['height']

        # getting current row crop name with extension
        crop_name_w_extension = f'{crop_name}{images_extension}'

        # getting current row crop paths
        red_path = join(red_folder, crop_name_w_extension)
        green_path = join(green_folder, crop_name_w_extension)

        # getting current crop red/green pixel intensity values
        red_intensity = get_nucleus_pixel_intensity(crop_path=red_path,
                                                    nucleus_width=nucleus_width,
                                                    nucleus_height=nucleus_height,
                                                    calc='median')
        green_intensity = get_nucleus_pixel_intensity(crop_path=green_path,
                                                      nucleus_width=nucleus_width,
                                                      nucleus_height=nucleus_height,
                                                      calc='median')

        # normalizing values
        red_intensity = red_intensity / 255
        green_intensity = green_intensity / 255

        # getting current crop cell cycle
        # current_class = get_cell_cycle(red_value=red_intensity,
        #                                green_value=green_intensity,
        #                                min_red_value=MIN_RED_VALUE,
        #                                min_green_value=MIN_GREEN_VALUE,
        #                                ratio_lower_threshold=RATIO_LOWER_THRESHOLD,
        #                                ratio_upper_threshold=RATIO_UPPER_THRESHOLD)
        current_class = get_cell_cycle_simple(red_value=red_intensity,
                                              green_value=green_intensity)

        # updating current row data
        df.at[row_index, col_name] = current_class

        # updating current row index
        current_row_index += 1


def generate_cell_cycle_annotations(crops_file: str,
                                    red_folder: str,
                                    green_folder: str,
                                    images_extension: str,
                                    output_path: str,
                                    ) -> None:
    """
    Given a path to a folder containing crops,
    and a path to a file containing crops info,
    generates cell cycle annotations, and saves
    it to given output path.
    """
    # reading crops info df
    crops_info_df = read_csv(crops_file)

    # updating class col
    add_cell_cycle_col(df=crops_info_df,
                       red_folder=red_folder,
                       green_folder=green_folder,
                       images_extension=images_extension)

    # saving updated crops info df
    crops_info_df.to_csv(output_path,
                         index=False)

    # printing execution message
    print(f'output saved to {output_path}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting crops file
    crops_file = args_dict['crops_file']

    # getting red folder
    red_folder = args_dict['red_folder']

    # getting green folder
    green_folder = args_dict['green_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting output path
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_cell_cycle_annotations function
    generate_cell_cycle_annotations(crops_file=crops_file,
                                    red_folder=red_folder,
                                    green_folder=green_folder,
                                    images_extension=images_extension,
                                    output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
