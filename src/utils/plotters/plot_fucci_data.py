# plot fucci cytometry module

print('initializing...')  # noqa

# Code destined to plotting red/green intensity
# cytometry-like plot, based on forNMA output file.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import DataFrame
from seaborn import scatterplot
from numpy import log2 as np_log2
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import add_cell_cycle_col
from src.utils.aux_funcs import create_analysis_df
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import add_cell_cycle_proportions_col
print('all required libraries successfully imported.')  # noqa

######################################################################
# defining global variables

IMAGE_NAME_COL = 'Image_name_merge'
MIN_RED_VALUE = 0.10
MIN_GREEN_VALUE = 0.05
RATIO_LOWER_THRESHOLD = 0.8
RATIO_UPPER_THRESHOLD = 1.2
TREATMENT_DICT = {'A1': 'TMZ',
                  'A2': 'CTR',
                  'A3': 'TMZ',
                  'A4': 'CTR',
                  'B1': 'TMZ',
                  'B2': 'CTR',
                  'B3': 'TMZ',
                  'B4': 'CTR',
                  'C1': 'TMZ',
                  'C2': 'CTR',
                  'C3': 'TMZ',
                  'C4': 'CTR'}

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "plot fucci histograms module"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # fornma file param
    parser.add_argument('-f', '--fornma-file',
                        dest='fornma_file',
                        required=True,
                        help='defines path to fornma output file (.csv)')

    # output folder param
    output_help = 'defines path to output folder'
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=False,
                        help=output_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_analysis_df(fornma_file_path: str,
                    image_name_col: str,
                    treatment_dict: dict,
                    min_red_value: float,
                    min_green_value: float,
                    ratio_lower_threshold: float,
                    ratio_upper_threshold: float
                    ) -> DataFrame:
    """
    Given a fornma file path,
    returns base analysis data frame.
    """
    # getting analysis df
    analysis_df = create_analysis_df(fornma_file_path=fornma_file_path,
                                     image_name_col=image_name_col,
                                     treatment_dict=treatment_dict)

    # dropping unrequired cols
    cols_to_keep = ['Cell',
                    'X',
                    'Y',
                    'Area',
                    'NII',
                    'Well',
                    'Field',
                    'Image_name_merge',
                    'Treatment',
                    'Mean_red',
                    'Mean_green']
    analysis_df = analysis_df[cols_to_keep]

    # adding red/green ratio col
    analysis_df['red_green_ratio'] = analysis_df['Mean_red'] / analysis_df['Mean_green']

    # adding red/green ratio log col
    analysis_df['red_green_ratio_log2'] = np_log2(analysis_df['red_green_ratio'])

    # adding cell cycle col
    add_cell_cycle_col(df=analysis_df,
                       min_red_value=min_red_value,
                       min_green_value=min_green_value,
                       ratio_lower_threshold=ratio_lower_threshold,
                       ratio_upper_threshold=ratio_upper_threshold)

    # adding cell cycle proportions col
    add_cell_cycle_proportions_col(df=analysis_df)

    # returning analysis df
    return analysis_df


def plot_cytometry(df: DataFrame,
                   output_folder: str
                   ) -> None:
    """
    Given an analysis data frame,
    plots cytometry-like plot,
    saving plot in given output folder.
    """
    # grouping df by treatment
    treatment_groups = df.groupby('Treatment')

    # iterating over treatment groups
    for treatment, treatment_group in treatment_groups:

        # defining save name/path
        save_name = f'{treatment}_fucci_cytometry.png'
        save_path = join(output_folder,
                         save_name)

        # setting figure size
        plt.figure(figsize=(14, 8))

        # getting total cells count
        total_cells_count = len(treatment_group)

        # getting current cell cycle proportions
        current_proportions = treatment_group['cell_cycle (%cells)'].unique()

        # sorting proportions so that it's always [G1, G2, M-eG1, S]
        sorted_proportions = sorted(current_proportions)

        # defining palette based on order [G1, G2, M-eG1, S]
        colors_list = ['red', 'green', 'tomato', 'yellow']

        # plotting figure
        scatterplot(data=treatment_group,
                    x='Mean_red',
                    y='Mean_green',
                    hue='cell_cycle (%cells)',
                    hue_order=sorted_proportions,
                    palette=colors_list)

        # setting plot title
        title = f'Fucci "Cytometry" (by mean pixel intensities) '
        title += f'| Treatment: {treatment} '
        title += f'| Total cells: {total_cells_count}'
        plt.title(title)

        # setting axes lims
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

        # setting figure layout
        plt.tight_layout()

        # saving figure
        plt.savefig(save_path)


def plot_ratio_log(df: DataFrame,
                   output_folder: str
                   ) -> None:
    """
    Given an analysis data frame,
    plots ratio log plot, saving plot
    in given output folder.
    """
    # grouping df by treatment
    treatment_groups = df.groupby('Treatment')

    # iterating over treatment groups
    for treatment, treatment_group in treatment_groups:

        # defining save name/path
        save_name = f'{treatment}_ratio_log.png'
        save_path = join(output_folder,
                         save_name)

        # setting figure size
        plt.figure(figsize=(14, 8))

        # getting total cells count
        total_cells_count = len(treatment_group)

        # getting current cell cycle proportions
        current_proportions = treatment_group['cell_cycle (%cells)'].unique()

        # sorting proportions so that it's always [G1, G2, M-eG1, S]
        sorted_proportions = sorted(current_proportions)

        # defining palette based on order [G1, G2, M-eG1, S]
        colors_list = ['red', 'green', 'tomato', 'yellow']

        # plotting figure
        scatterplot(data=treatment_group,
                    x='red_green_ratio_log2',
                    y='Area',
                    hue='cell_cycle (%cells)',
                    hue_order=sorted_proportions,
                    palette=colors_list)

        # setting plot title
        title = f'Fucci log(red_mean/green_mean) '
        title += f'| Treatment: {treatment} '
        title += f'| Total cells: {total_cells_count}'
        plt.title(title)

        # setting figure layout
        plt.tight_layout()

        # saving figure
        plt.savefig(save_path)


def plot_ratio_log(df: DataFrame,
                   output_folder: str
                   ) -> None:
    """
    Given an analysis data frame,
    plots ratio log plot, saving plot
    in given output folder.
    """
    # grouping df by treatment
    treatment_groups = df.groupby('Treatment')

    # iterating over treatment groups
    for treatment, treatment_group in treatment_groups:

        # defining save name/path
        save_name = f'{treatment}_ratio_log.png'
        save_path = join(output_folder,
                         save_name)

        # setting figure size
        plt.figure(figsize=(14, 8))

        # getting total cells count
        total_cells_count = len(treatment_group)

        # getting current cell cycle proportions
        current_proportions = treatment_group['cell_cycle (%cells)'].unique()

        # sorting proportions so that it's always [G1, G2, M-eG1, S]
        sorted_proportions = sorted(current_proportions)

        # defining palette based on order [G1, G2, M-eG1, S]
        colors_list = ['red', 'green', 'tomato', 'yellow']

        # plotting figure
        scatterplot(data=treatment_group,
                    x='red_green_ratio_log2',
                    y='Area',
                    hue='cell_cycle (%cells)',
                    hue_order=sorted_proportions,
                    palette=colors_list)

        # setting plot title
        title = f'Fucci log(red_mean/green_mean) '
        title += f'| Treatment: {treatment} '
        title += f'| Total cells: {total_cells_count}'
        plt.title(title)

        # setting figure layout
        plt.tight_layout()

        # saving figure
        plt.savefig(save_path)


def plot_fucci_cytometry(fornma_file_path: str,
                         image_name_col: str,
                         treatment_dict: dict,
                         output_folder: str
                         ) -> None:
    """
    Given paths to a fornma output file,
    plots cytometry-like plot, based on
    red/green channels intensities.
    """
    # getting analysis df
    print('getting analysis df...')
    analysis_df = get_analysis_df(fornma_file_path=fornma_file_path,
                                  image_name_col= image_name_col,
                                  treatment_dict=treatment_dict,
                                  min_red_value=MIN_RED_VALUE,
                                  min_green_value=MIN_GREEN_VALUE,
                                  ratio_lower_threshold=RATIO_LOWER_THRESHOLD,
                                  ratio_upper_threshold=RATIO_UPPER_THRESHOLD)

    # plotting cytometry plot
    print('plotting cytometry-like plot...')
    plot_cytometry(df=analysis_df,
                   output_folder=output_folder)

    # plotting ratio log plot
    print('plotting ratio log plot...')
    plot_ratio_log(df=analysis_df,
                   output_folder=output_folder)

    # printing execution message
    print('analysis complete.')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting fornma file path
    fornma_file = args_dict['fornma_file']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running plot_fucci_cytometry function
    plot_fucci_cytometry(fornma_file_path=fornma_file,
                         image_name_col=IMAGE_NAME_COL,
                         treatment_dict=TREATMENT_DICT,
                         output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
