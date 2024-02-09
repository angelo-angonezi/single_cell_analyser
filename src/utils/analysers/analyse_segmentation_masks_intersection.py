# analyse segmentation masks intersection module

print('initializing...')  # noqa

# Code destined to analysing segmentation
# masks intersections based on BRAIND detections.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from numpy import arange
from os.path import join
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from seaborn import lineplot
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_segmentation_mask
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

######################################################################
# defining global variables

ER_MIN = 1.0
ER_MAX = 4.0
ER_STEP = 0.2

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'analyse segmentation masks intersections module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # detection file param
    parser.add_argument('-d', '--detection_file',
                        dest='detection_file',
                        required=True,
                        help='defines path to csv file containing model detections')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to output folder')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_single_er_df(df: DataFrame,
                     er: float
                     ) -> DataFrame:
    """
    Given a detections data frame,
    and an expansion ratio value,
    returns OBBs intersection df.
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # grouping df by image
    image_groups = df.groupby('img_file_name')

    # iterating over image groups
    for image_name, image_group in image_groups:

        # generating segmentation mask for current image
        segmentation_mask = get_segmentation_mask(df=image_group,
                                                  style='ellipse',
                                                  expansion_ratio=er)

        # getting current image/er intersection pixels count
        intersection_pixels = segmentation_mask[segmentation_mask > 1]
        intersection_pixels_count = len(intersection_pixels)

        # assembling current image/er dict
        current_dict = {'image_name': image_name,
                        'er': er,
                        'intersection_pixels_count': intersection_pixels_count}

        # assembling current image/er df
        current_df = DataFrame(current_dict,
                               index=[0])

        # appending current image/er df to dfs list
        dfs_list.append(current_df)

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def get_ers_df(df: DataFrame,
               er_min: float,
               er_max: float,
               er_step: float
               ) -> DataFrame:
    """
    Given a detections data frame,
    and parameters for expansion ratio
    range, returns masks intersections
    analysis data frame.
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # updating max value
    er_max += er_step

    # getting expansion ratio range
    ers_range = arange(start=er_min,
                       stop=er_max,
                       step=er_step)

    # getting expansion ratio list
    ers = [round(er, 2) for er in ers_range]
    ers_num = len(ers)

    # defining starter for current_er_index
    current_er_index = 1

    # iterating over er in er list
    for er in ers:

        # printing progress message
        base_string = f'calculating obbs intersection for er: {er} | #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_er_index,
                               total=ers_num)

        # getting current er df
        current_er_df = get_single_er_df(df=df,
                                         er=er)

        # appending current er df to dfs list
        dfs_list.append(current_er_df)

        # updating current_er_index
        current_er_index += 1

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def plot_obbs_intersection(df: DataFrame,
                           output_folder: str
                           ) -> None:
    """
    Given an ers data frame,
    plots data and saves it in
    given output folder.
    """
    # defining save name/path
    save_name = f'intersections_plot.png'
    save_path = join(output_folder,
                     save_name)

    # setting figure size
    plt.figure(figsize=(14, 8))

    # plotting figure
    lineplot(data=df,
             x='er',
             y='intersection_pixels_count',
             hue='image_name')

    # setting plot title
    title = f'OBBs intersections plot'
    plt.title(title)

    # setting figure layout
    plt.tight_layout()

    # saving figure
    plt.savefig(save_path)


def analyse_segmentation_masks_intersections(detections_file: str,
                                             output_folder: str
                                             ) -> None:
    """
    Given paths to model detections file,
    creates analysis data frames and plots
    to assess intersection between OBBs in
    increasing expansion ratios.
    """
    # reading detections file
    print('reading detections file...')
    detections_df = read_csv(detections_file)

    # getting ers df
    print('getting ers df...')
    ers_df = get_ers_df(df=detections_df,
                        er_min=ER_MIN,
                        er_max=ER_MAX,
                        er_step=ER_STEP)

    # saving ers df
    print('saving ers df...')
    save_name = 'ers_df.csv'
    save_path = join(output_folder,
                     save_name)
    ers_df.to_csv(save_path,
                  index=False)

    # plotting data
    print('plotting data...')
    plot_obbs_intersection(df=ers_df,
                           output_folder=output_folder)

    # printing execution message
    print(f'output saved to {output_folder}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting detections file
    detections_file = args_dict['detection_file']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_autophagy_dfs function
    analyse_segmentation_masks_intersections(detections_file=detections_file,
                                             output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
