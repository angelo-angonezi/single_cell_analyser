# check pixel intensities module

print('initializing...')  # noqa

# Code destined to plotting pixel
# intensities (min/max/mean) to compare
# incucyte images from same experiment.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from cv2 import imread
from os.path import join
from numpy import ndarray
from os.path import exists
from pandas import read_csv
from seaborn import barplot
from pandas import DataFrame
from pandas import read_pickle
from keras.models import Model
from keras.utils import load_img
from sklearn.cluster import KMeans
from plotly.express import scatter
from numpy import array as np_array
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from os import makedirs as os_makedirs
from src.utils.aux_funcs import get_base_df
from src.utils.aux_funcs import print_gpu_usage
from sklearn.preprocessing import StandardScaler
from src.utils.aux_funcs import add_file_path_col
from src.utils.aux_funcs import enter_to_continue
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
    description = "check pixel intensities module"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    input_help = 'defines input folder (folder containing images)'
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help=input_help)

    # image extension param
    extension_help = 'defines extension (.tif, .png, .jpg) of images in input folder'
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help=extension_help)

    # output folder param
    output_help = 'defines folder which will contain output plots'
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


def get_pixel_intensities(file_path: str) -> DataFrame:
    """
    Given a path to an image,
    returns tuple of mean/min/max
    pixel intensity values.
    """
    # loading image
    open_img = imread(file_path,
                      -1)  # reads image as is (independent on input format)

    # getting image mean/min/max values
    pixel_mean = open_img.mean()
    pixel_min = open_img.min()
    pixel_max = open_img.max()

    # converting values to ints
    pixel_mean = int(pixel_mean)
    pixel_min = int(pixel_min)
    pixel_max = int(pixel_max)

    # assembling pixel intensities tuple
    intensities_tuple = (pixel_mean, pixel_min, pixel_max)

    # returning intensities tuple
    return intensities_tuple


def add_pixel_intensities_cols(df: DataFrame) -> None:
    """
    Given a base data frame, adds
    mean/min/max pixel intensity
    columns.
    """
    # defining new cols names
    mean_col = 'mean'
    min_col = 'min'
    max_col = 'max'

    # defining placeholder values for new cols
    df[mean_col] = None
    df[min_col] = None
    df[max_col] = None

    # getting df rows
    df_rows = df.iterrows()

    # getting rows num
    rows_num = len(df)

    # defining starter for current row index
    current_row_index = 1

    # iterating over rows
    for row in df_rows:

        # printing progress message
        base_string = f'adding pixel intensities cols (row #INDEX# of #TOTAL#)'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row index/data
        row_index, row_data = row

        # getting current row image path
        file_path = row_data['file_path']

        # getting current image pixel intensities
        current_mean, current_min, current_max = get_pixel_intensities(file_path=file_path)

        # updating current row new cols values
        df.at[row_index, mean_col] = current_mean
        df.at[row_index, min_col] = current_min
        df.at[row_index, max_col] = current_max

        # updating current row index
        current_row_index += 1


def create_pixel_intensities_df(input_folder: str,
                                images_extension: str
                                ) -> DataFrame:
    """
    Creates pixel intensities data frame, based
    on images in given input folder.
    """
    # getting files in input folder
    files = get_specific_files_in_folder(path_to_folder=input_folder,
                                         extension=images_extension)

    # creating df
    intensities_df = get_base_df(files=files,
                                 col_name='file_name')

    # adding file path col
    add_file_path_col(df=intensities_df,
                      input_folder=input_folder)

    # adding intensities cols
    add_pixel_intensities_cols(df=intensities_df)

    # dropping file path col
    intensities_df = intensities_df[['file_name', 'mean', 'min', 'max']]

    # melting df
    intensities_df = intensities_df.melt(id_vars=['file_name'])

    # returning intensities df
    return intensities_df


def get_pixel_intensities_df(input_folder: str,
                             images_extension: str,
                             output_folder: str
                             ) -> DataFrame:
    """
    Checks whether pixel intensities data frame
    exists in given output folder, and returns
    loaded df. Creates it from scratch otherwise.
    """
    # defining placeholder value for intensities df
    intensities_df = None

    # defining file name
    file_name = 'pixel_intensities_df.pickle'

    # getting file path
    file_path = join(output_folder,
                     file_name)

    # checking whether file exists
    if exists(file_path):

        # loading intensities df from existing file
        print('loading pixel intensities df from existing file...')
        intensities_df = read_pickle(file_path)

    # if it does not exist
    else:

        # creating intensities df
        print('creating pixel intensities df from scratch...')
        intensities_df = create_pixel_intensities_df(input_folder=input_folder,
                                                     images_extension=images_extension)

        # saving intensities df
        print('saving intensities df...')
        intensities_df.to_pickle(file_path)

    # returning intensities df
    return intensities_df


def plot_intensities(df: DataFrame,
                     input_folder: str,
                     output_folder: str
                     ) -> None:
    """
    Given an intensities data frame,
    saves plots in given output folder.
    """
    # defining save name/path
    save_name = 'intensities_plot.png'
    save_path = join(output_folder,
                     save_name)

    # getting folder name
    folder_split = input_folder.split('/')
    folder_name = folder_split[-2]

    # setting figure size
    plt.figure(figsize=(14, 8))

    # plotting figure
    fig = barplot(data=df,
                  x='file_name',
                  y='value',
                  hue='variable')

    # adding labels to bars
    for i in range(len(df['variable'].unique())):
        fig.bar_label(fig.containers[i], label_type='edge')

    # rotating x-ticks
    plt.xticks(rotation=30)

    # setting plot title
    plt.title(folder_name)

    # setting figure layout
    plt.tight_layout()

    # saving figure
    plt.savefig(save_path)


def check_pixel_intensities(input_folder: str,
                            images_extension: str,
                            output_folder: str
                            ) -> None:
    """
    Given a path to a folder containing images,
    checks pixel intensities based on mean/min/max
    values, saving resulting plots in given output
    folder.
    """
    # getting pixel intensity df
    print('getting pixel intensity df...')
    pixel_intensity_df = get_pixel_intensities_df(input_folder=input_folder,
                                                  images_extension=images_extension,
                                                  output_folder=output_folder)

    # plotting data
    print('plotting data...')
    plot_intensities(df=pixel_intensity_df,
                     input_folder=input_folder,
                     output_folder=output_folder)

    # printing execution message
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

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running check_pixel_intensities function
    check_pixel_intensities(input_folder=input_folder,
                            images_extension=images_extension,
                            output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
