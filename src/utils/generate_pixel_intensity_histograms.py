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
from pandas import read_csv
from pandas import DataFrame
from seaborn import histplot
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

        # TODO: remove this once test completed
        # skipping to next
        dfs_list.append(image_group)
        continue

        # getting current image red/green dfs
        red_df = image_group[image_group['channel'] == 'red']
        green_df = image_group[image_group['channel'] == 'green']

        # getting current image red/green pixels
        red_pixels = red_df['pixel_intensity']
        green_pixels = green_df['pixel_intensity']

        # getting current image red/green min/max values
        red_min = red_pixels.min()
        red_max = red_pixels.max()
        green_min = green_pixels.min()
        green_max = green_pixels.max()

        print(green_pixels)
        print(red_min)
        # TODO: check this logic
        # normalizing pixels by min values
        red_pixels -= red_min
        green_pixels = green_pixels - green_min

        print(green_pixels)
        exit()

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
