# generate pixel intensity histograms module

print('initializing...')  # noqa

# Code destined to generating pixel intensity
# histograms - Fucci related.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from numpy import ndarray
from pandas import read_csv
from pandas import DataFrame
from seaborn import histplot
from numpy import min as np_min
from numpy import max as np_max
from numpy import all as np_all
from argparse import ArgumentParser
from matplotlib import pyplot as plt
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
    description = 'generate pixel intensity histograms for single-nuclei crops'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # red folder param
    parser.add_argument('-d', '--crops-pixels-df',
                        dest='crops_pixels_df',
                        required=True,
                        help='defines path to file containing info on crops pixels (.csv)')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines output folder (folder that will contain output files)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_pixels_df(crops_file: str) -> DataFrame:
    """
    Given a path to a crops info csv,
    returns crops data frame.
    :param crops_file: String. Represents a path to a file.
    :return: DataFrame. Represents crops pixels data frame.
    """
    # defining col types
    col_types = {'img_name': str,
                 'crop_name': str,
                 'channel': str,
                 'pixel_intensity': int}

    # reading crops file
    crops_df = read_csv(crops_file,
                        dtype=col_types)

    # returning crops df
    return crops_df


def get_normalized_array(arr: ndarray) -> ndarray:
    """
    Given a pixels' array, returns array
    normalized by min/max pixel values.
    """
    # getting array min/max values
    arr_min = np_min(arr)
    arr_max = np_max(arr)

    # TODO: check this logic!
    # subtracting min value from all elements in array (lower normalization)
    arr = arr - arr_min

    # dividing all elements in array by max value (upper normalization)
    arr = arr / arr_max

    # returning normalized array
    return arr


def get_normalized_df(df: DataFrame) -> DataFrame:
    """
    Given a crops pixels data frame,
    returns a normalized copy of given
    df, in which pixels have been adjusted
    according to image minimum/maximum
    (for each channel).
    :param df: DataFrame. Represents a crops pixels data frame.
    :return: DataFrame. Represents a normalized pixels data frame.
    """
    # defining placeholder value for dfs_list
    dfs_list = []

    # grouping df by images
    image_groups = df.groupby('img_name')

    # getting images num
    images_num = len(image_groups)

    # defining start value for current_img_index
    current_img_index = 1

    # iterating over image groups
    for image_name, image_group in image_groups:

        # printing execution message
        f_string = f'normalizing values for image #INDEX# of #TOTAL#'
        print_progress_message(base_string=f_string,
                               index=current_img_index,
                               total=images_num)

        # updating index
        current_img_index += 1

        # getting current image red/green dfs
        red_df = image_group[image_group['channel'] == 'red']
        green_df = image_group[image_group['channel'] == 'green']

        # getting current image red/green pixels (as numpy arrays)
        red_pixels = red_df['pixel_intensity'].to_numpy()
        green_pixels = green_df['pixel_intensity'].to_numpy()

        # normalizing arrays
        red_pixels_normalized = get_normalized_array(arr=red_pixels)
        green_pixels_normalized = get_normalized_array(arr=green_pixels)

        # getting image names
        image_names = image_group['img_name']

        # getting crop names
        crop_names = image_group['crop_name']

        #  assembling current image dict
        red_list = ['red' for _ in red_pixels]
        green_list = ['green' for _ in green_pixels]
        red_list.extend(green_list)
        red_pixels_list = [pixel for pixel in red_pixels_normalized]
        green_pixels_list = [pixel for pixel in green_pixels_normalized]
        red_pixels_list.extend(green_pixels_list)
        current_dict = {'img_name': image_names,
                        'crop_name': crop_names,
                        'channel': red_list,
                        'pixel_intensity': red_pixels_list}

        # assembling current image df
        current_df = DataFrame(current_dict)

        # appending current image df to dfs_list
        dfs_list.append(current_df)

    # concatenating dfs in dfs_list
    final_df = concat(dfs_list)

    # returning final df
    return final_df


def plot_pixel_histograms(df: DataFrame,
                          output_folder: str
                          ) -> None:
    """
    Given a normalized pixels data frame,
    creates pixel intensity histograms,
    saving output to given folder.
    :param df: DataFrame. Represents a normalized pixels data frame.
    :param output_folder: String. Represents a path to a folder.
    :return: None.
    """
    # grouping df by crop
    crop_groups = df.groupby('crop_name')

    # getting crops num
    crops_num = len(crop_groups)

    # defining start value for current_crop_index
    current_crop_index = 1

    # iterating over crop groups
    for crop_name, crop_group in crop_groups:

        # printing execution message
        f_string = f'generating histograms for crop #INDEX# of #TOTAL#'
        print_progress_message(base_string=f_string,
                               index=current_crop_index,
                               total=crops_num)

        # updating index
        current_crop_index += 1

        # generating current crop pixel pairs histogram
        histplot(data=crop_group,
                 x='pixel_intensity',
                 hue='channel',
                 hue_order=['red', 'green'],
                 palette=['r', 'g'],
                 kde=False)

        # saving plot
        save_name = f'{crop_name}_histograms.png'
        save_path = join(output_folder,
                         save_name)
        plt.savefig(save_path)

        # closing plot
        plt.close()


def generate_pixel_intensity_histograms(crops_file: str,
                                        output_folder: str,
                                        ) -> None:
    """
    Given a path to a folder containing crops,
    and a path to a file containing crops info,
    generates ML input compatible tables
    :param crops_file: String. Represents a path to a file.
    :param output_folder: String. Represents a path to a folder.
    :return: None.
    """
    # reading crops pixels df
    crops_pixels_df = get_pixels_df(crops_file=crops_file)

    # normalizing pixel intensities
    normalized_pixels_df = get_normalized_df(df=crops_pixels_df)

    # generating plots
    plot_pixel_histograms(df=normalized_pixels_df,
                          output_folder=output_folder)

    # printing execution message
    print(f'files saved to {output_folder}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting crops pixels file
    crops_file = args_dict['crops_pixels_df']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running generate_pixel_intensity_histograms function
    generate_pixel_intensity_histograms(crops_file=crops_file,
                                        output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
