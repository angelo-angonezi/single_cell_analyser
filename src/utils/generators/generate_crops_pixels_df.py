# generate crops pixels df module

print('initializing...')  # noqa

# Code destined to generating crop pixels
# data frame - Fucci related.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import get_crops_df
from src.utils.aux_funcs import get_crop_pixels
from src.utils.aux_funcs import enter_to_continue
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
    description = 'generate crops pixels df module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

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

    # crops file param
    crops_help = 'defines path to crops file (containing crops info)'
    parser.add_argument('-c', '--crops-file',
                        dest='crops_file',
                        required=True,
                        help=crops_help)

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


def get_crops_pixels_df(red_folder: str,
                        green_folder: str,
                        images_extension: str,
                        crops_file: str
                        ) -> DataFrame:
    """
    Given a path to a folder containing crops,
    and a path to a file containing crops info,
    generates crops pixels data frame, and saves
    it to given output path.
    :param red_folder: String. Represents a path to a folder.
    :param green_folder: String. Represents a path to a folder.
    :param images_extension: String. Represents image extension.
    :param crops_file: String. Represents a path to a file.
    :return: DataFrame. Represents a crops pixels data frame.
    """
    # getting crops df
    crops_df = get_crops_df(crops_file=crops_file)

    # getting crops num
    crops_num = len(crops_df)

    # defining placeholder value for dfs_list
    dfs_list = []

    # getting df rows
    df_rows = crops_df.iterrows()

    # defining start value for current_crop_index
    current_crop_index = 1

    # iterating over df rows
    for row_index, row_data in df_rows:

        # printing execution message
        f_string = f'getting pixels from crop #INDEX# of #TOTAL#'
        print_progress_message(base_string=f_string,
                               index=current_crop_index,
                               total=crops_num)

        # updating index
        current_crop_index += 1

        # getting current crop image name
        crop_img_name = row_data['img_name']

        # getting current crop name
        crop_name = row_data['crop_name']
        crop_name_w_extension = f'{crop_name}{images_extension}'

        # getting current crop paths
        red_path = join(red_folder, crop_name_w_extension)
        green_path = join(green_folder, crop_name_w_extension)

        # getting current crop pixels
        red_pixels = get_crop_pixels(crop_path=red_path)
        green_pixels = get_crop_pixels(crop_path=green_path)

        # getting pixel range list
        pixels_num = len(red_pixels)
        pixel_range = range(pixels_num)
        pixel_ids = [i for i in pixel_range]

        # getting names list
        crop_name_list = [crop_name for _ in red_pixels]
        img_name_list = [crop_img_name for _ in red_pixels]

        # assembling current crop pair dict
        current_dict = {'img_name': img_name_list,
                        'crop_name': crop_name_list,
                        'pixel': pixel_ids,
                        'red': red_pixels,
                        'green': green_pixels}

        # assembling current crop pair df
        current_df = DataFrame(current_dict)

        # appending current df to dfs_list
        dfs_list.append(current_df)

    # concatenating dfs in dfs_list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning df
    return final_df


def get_normalized_pixel(pixel_value: int,
                         arr_min: int,
                         arr_max: int
                         ) -> float:
    """
    Given a pixel value, and min/max values
    for pixels in original array, returns
    pixel normalized value.
    :param pixel_value: Integer. Represents a pixel value.
    :param arr_min: Integer. Represents a pixel value.
    :param arr_max: Integer. Represents a pixel value.
    :return: Float. Represents normalized pixel value.
    """
    # TODO: check this normalization logic!
    # normalizing value by lower threshold
    pixel_value_normalized = pixel_value - arr_min

    # normalizing value by upper threshold
    pixel_value_normalized = pixel_value_normalized / arr_max

    # returning normalized value
    return pixel_value_normalized


def add_normalized_columns(df: DataFrame) -> None:
    """
    Given a crops pixels data frame,
    adds red/green pixels normalized
    columns, based on min/max pixel
    values for each image.
    :param df: DataFrame. Represents a crops pixels data frame.
    :return: None.
    """
    # defining column names
    red_normalized_col = 'red_normalized'
    green_normalized_col = 'green_normalized'

    # defining placeholder value for columns
    df[red_normalized_col] = None
    df[green_normalized_col] = None

    # grouping df by images
    image_groups = df.groupby('img_name')

    # getting rows num
    rows_num = len(df)

    # defining starter for current_row_index
    current_row_index = 1

    # iterating over image groups
    for image_name, image_group in image_groups:

        # getting current image red/green pixels
        red_pixels = image_group['red']
        green_pixels = image_group['green']

        # getting current image min/max values
        red_min = red_pixels.min()
        red_max = red_pixels.max()
        green_min = green_pixels.min()
        green_max = green_pixels.max()

        # getting current image rows
        image_rows = image_group.iterrows()

        # iterating over image rows
        for row_index, row_data in image_rows:

            # printing execution message
            base_string = 'normalizing pixel #INDEX# of #TOTAL#'
            print_progress_message(base_string=base_string,
                                   index=current_row_index,
                                   total=rows_num)

            # updating current_row_index
            current_row_index += 1

            # getting current row red/green pixel values
            red_value = row_data['red']
            green_value = row_data['green']

            # getting normalized pixel values
            red_value_normalized = get_normalized_pixel(pixel_value=red_value,
                                                        arr_min=red_min,
                                                        arr_max=red_max)
            green_value_normalized = get_normalized_pixel(pixel_value=green_value,
                                                          arr_min=green_min,
                                                          arr_max=green_max)

            # updating values in respective columns
            df.at[row_index, red_normalized_col] = red_value_normalized
            df.at[row_index, green_normalized_col] = green_value_normalized


def generate_crops_pixels_df(red_folder: str,
                             green_folder: str,
                             images_extension: str,
                             crops_file: str,
                             output_path: str,
                             ) -> None:
    """
    Given a path to a folder containing crops,
    and a path to a file containing crops info,
    generates crops pixels data frame, and saves
    it to given output path.
    :param red_folder: String. Represents a path to a folder.
    :param green_folder: String. Represents a path to a folder.
    :param images_extension: String. Represents image extension.
    :param crops_file: String. Represents a path to a file.
    :param output_path: String. Represents a path to a file.
    :return: None.
    """
    # getting crops pixels df
    crops_pixels_df = get_crops_pixels_df(red_folder=red_folder,
                                          green_folder=green_folder,
                                          images_extension=images_extension,
                                          crops_file=crops_file)

    # normalizing pixels in df
    add_normalized_columns(df=crops_pixels_df)

    # saving crops pixels df
    crops_pixels_df.to_csv(output_path,
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

    # getting red folder
    red_folder = args_dict['red_folder']

    # getting green folder
    green_folder = args_dict['green_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting crops file
    crops_file = args_dict['crops_file']

    # getting output path
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_crops_pixels_df function
    generate_crops_pixels_df(red_folder=red_folder,
                             green_folder=green_folder,
                             images_extension=images_extension,
                             crops_file=crops_file,
                             output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
