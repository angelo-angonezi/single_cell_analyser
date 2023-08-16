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
    crops_pixels_df = read_csv(crops_file)

    # generating current crop pair histogram
    # TODO: add grouping/loop before running this part
    histplot(data=crops_pixels_df,
             x='pixel_intensity',
             hue='channel',
             hue_order=['red', 'green'],
             palette=['r', 'g'],
             kde=False)

    # saving plot
    save_name = f'crop.png'
    save_path = join(output_folder,
                     save_name)
    plt.savefig(save_path)

    # closing plot
    plt.close()

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
    enter_to_continue()

    # running generate_pixel_intensity_histograms function
    generate_pixel_intensity_histograms(crops_file=crops_file,
                                        output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
