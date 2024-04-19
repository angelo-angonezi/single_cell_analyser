# plot fucci data module

print('initializing...')  # noqa

# Code destined to plotting red/green intensity
# cytometry-like plot, based on forNMA output file.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from os.path import exists
from pandas import read_csv
from pandas import DataFrame
from seaborn import scatterplot
from numpy import log2 as np_log2
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import get_analysis_df
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import add_cell_cycle_col
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import add_cell_cycle_proportions_col
print('all required libraries successfully imported.')  # noqa

######################################################################
# defining global variables

IMAGE_NAME_COL = 'Image_name_merge'
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
    description = "plot fucci data module"

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
                        required=True,
                        help=output_help)

    # treatments file param
    parser.add_argument('-tr', '--treatment-file',
                        dest='treatment_file',
                        help='defines path to file containing treatment info',
                        required=True)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def create_fucci_df(fornma_file_path: str,
                    image_name_col: str,
                    treatment_file: str,
                    min_red_value: float,
                    min_green_value: float,
                    ratio_lower_threshold: float,
                    ratio_upper_threshold: float,
                    output_folder: str
                    ) -> DataFrame:
    """
    Given a fornma file path, returns
    df with added columns for further
    cell cycle analysis.
    """
    # getting analysis df
    analysis_df = get_analysis_df(base_df_path=fornma_file_path,
                                  image_name_col=image_name_col,
                                  output_folder=output_folder,
                                  treatment_file=treatment_file)

    # dropping unrequired cols
    cols_to_keep = ['Cell',
                    'X',
                    'Y',
                    'Area',
                    'NII',
                    'well',
                    'field',
                    'date',
                    'time',
                    'Image_name_merge',
                    'treatment',
                    'Mean_red',
                    'Mean_green']
    analysis_df = analysis_df[cols_to_keep]

    # adding red/green ratio col
    analysis_df['red_green_ratio'] = analysis_df['Mean_red'] / analysis_df['Mean_green']

    # adding red/green ratio log col
    analysis_df['red_green_ratio_log2'] = np_log2(analysis_df['red_green_ratio'])

    # adding area log col
    analysis_df['area_log2'] = np_log2(analysis_df['Area'])

    # adding red+green log col
    analysis_df['red_and_green'] = analysis_df['Mean_red'] + analysis_df['Mean_green']

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


def get_fucci_df(fornma_file_path: str,
                 image_name_col: str,
                 treatment_file: str,
                 min_red_value: float,
                 min_green_value: float,
                 ratio_lower_threshold: float,
                 ratio_upper_threshold: float,
                 output_folder: str
                 ) -> DataFrame:
    """
    Checks whether fucci df already exists in
    output folder, loads and returns it.
    Creates it from scratch otherwise.
    """
    # defining csv output path
    save_name = 'fucci_df.csv'
    save_path = join(output_folder,
                     save_name)

    # defining placeholder value for fucci_df
    fucci_df = None

    # checking if csv output already exists
    if exists(save_path):

        # reading already existent data frame
        print('reading already existent data frame...')
        fucci_df = read_csv(save_path)

    # if output csv does not already exist
    else:

        # creating analysis_df
        print('creating fucci df...')
        fucci_df = create_fucci_df(fornma_file_path=fornma_file_path,
                                   image_name_col=image_name_col,
                                   treatment_file=treatment_file,
                                   min_red_value=min_red_value,
                                   min_green_value=min_green_value,
                                   ratio_lower_threshold=ratio_lower_threshold,
                                   ratio_upper_threshold=ratio_upper_threshold,
                                   output_folder=output_folder)

        # saving output csv
        print('saving fucci df...')
        fucci_df.to_csv(save_path,
                        index=False)

    # returning analysis_df
    return fucci_df


def plot_cytometry(df: DataFrame,
                   treatment: str,
                   date: str,
                   time: str,
                   total_cells: int,
                   proportions: list,
                   colors: list,
                   output_folder: str
                   ) -> None:
    """
    Given an analysis data frame,
    plots cytometry-like plot,
    saving plot to given output path.
    """
    # defining save name/path
    save_name = f'cytometry_{treatment}_{date}_{time}.png'
    save_path = join(output_folder,
                     save_name)

    # setting figure size
    plt.figure(figsize=(14, 8))

    # plotting figure
    scatterplot(data=df,
                x='Mean_red',
                y='Mean_green',
                hue='cell_cycle (%cells)',
                hue_order=proportions,
                palette=colors)

    # setting plot title
    title = f'Fucci "Cytometry" (by mean pixel intensities) '
    title += f'| Treatment: {treatment} '
    title += f'| Total cells: {total_cells}'
    plt.title(title)

    # setting axes lims
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    # setting figure layout
    plt.tight_layout()

    # saving figure
    plt.savefig(save_path)

    # closing figure
    plt.close()


def plot_ratio_log(df: DataFrame,
                   treatment: str,
                   date: str,
                   time: str,
                   total_cells: int,
                   proportions: list,
                   colors: list,
                   output_folder: str
                   ) -> None:
    """
    Given an analysis data frame,
    plots ratio log plot, saving plot
    to given output path.
    """
    # defining save name/path
    save_name = f'ratio_{treatment}_{date}_{time}.png'
    save_path = join(output_folder,
                     save_name)

    # setting figure size
    plt.figure(figsize=(14, 8))

    # plotting figure
    scatterplot(data=df,
                x='red_green_ratio_log2',
                y='area_log2',
                # y='red_and_green',
                hue='cell_cycle (%cells)',
                hue_order=proportions,
                palette=colors)

    # setting plot title
    title = f'Fucci log(red_mean/green_mean) '
    title += f'| Treatment: {treatment} '
    title += f'| Total cells: {total_cells}'
    plt.title(title)

    # setting figure layout
    plt.tight_layout()

    # saving figure
    plt.savefig(save_path)

    # closing figure
    plt.close()


def plot_fucci_nma(df: DataFrame,
                   treatment: str,
                   date: str,
                   time: str,
                   total_cells: int,
                   proportions: list,
                   colors: list,
                   output_folder: str
                   ) -> None:
    """
    Given an analysis data frame,
    plots fucci NMA plot, saving plot
    to given output path.
    """
    # defining save name/path
    save_name = f'nma_{treatment}_{date}_{time}.png'
    save_path = join(output_folder,
                     save_name)

    # setting figure size
    plt.figure(figsize=(14, 8))

    # plotting figure
    scatterplot(data=df,
                x='NII',
                y='Area',
                hue='cell_cycle (%cells)',
                hue_order=proportions,
                palette=colors)

    # setting plot title
    title = f'Fucci NMA '
    title += f'| Treatment: {treatment} '
    title += f'| Total cells: {total_cells}'
    plt.title(title)

    # setting axes lims
    # plt.xlim(0.0, 60)
    plt.ylim(0.0, 10000)

    # setting figure layout
    plt.tight_layout()

    # saving figure
    plt.savefig(save_path)

    # closing figure
    plt.close()


def generate_fucci_plots(df: DataFrame,
                         output_folder: str
                         ) -> None:
    """
    Given a fucci analysis df, groups
    data according to treatment and datetime,
    generating multiple plots in given output
    folder.
    """
    # defining group cols
    group_cols = ['treatment', 'date', 'time']

    # grouping df
    df_groups = df.groupby(group_cols)

    # iterating over df groups
    for df_name, df_group in df_groups:

        # getting current group info
        current_treatment, current_date, current_time = df_name

        # getting total cells count
        total_cells_count = len(df_group)

        # getting current cell cycle proportions
        current_proportions = df_group['cell_cycle (%cells)'].unique()

        # sorting proportions so that it's always [G1, G2, M-eG1, S]
        sorted_proportions = sorted(current_proportions)

        # defining palette based on order [G1, G2, M-eG1, S]
        colors_list = ['red', 'green', 'magenta', 'yellow']

        # plotting cytometry plot
        plot_cytometry(df=df_group,
                       treatment=current_treatment,
                       date=current_date,
                       time=current_time,
                       total_cells=total_cells_count,
                       proportions=sorted_proportions,
                       colors=colors_list,
                       output_folder=output_folder)

        # plotting ratio log plot
        plot_ratio_log(df=df_group,
                       treatment=current_treatment,
                       date=current_date,
                       time=current_time,
                       total_cells=total_cells_count,
                       proportions=sorted_proportions,
                       colors=colors_list,
                       output_folder=output_folder)

        # plotting fucci NMA plot
        plot_fucci_nma(df=df_group,
                       treatment=current_treatment,
                       date=current_date,
                       time=current_time,
                       total_cells=total_cells_count,
                       proportions=sorted_proportions,
                       colors=colors_list,
                       output_folder=output_folder)


def plot_fucci_cytometry(fornma_file_path: str,
                         image_name_col: str,
                         treatment_file: str,
                         output_folder: str
                         ) -> None:
    """
    Given paths to a fornma output file,
    plots cytometry-like plot, based on
    red/green channels intensities.
    """
    # getting analysis df
    print('getting fucci analysis df...')
    fucci_df = get_fucci_df(fornma_file_path=fornma_file_path,
                            image_name_col=image_name_col,
                            treatment_file=treatment_file,
                            min_red_value=MIN_RED_VALUE,
                            min_green_value=MIN_GREEN_VALUE,
                            ratio_lower_threshold=RATIO_LOWER_THRESHOLD,
                            ratio_upper_threshold=RATIO_UPPER_THRESHOLD,
                            output_folder=output_folder)

    # generating plots
    print('generating plots...')
    generate_fucci_plots(df=fucci_df,
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

    # getting treatment file path
    treatment_file = args_dict['treatment_file']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running plot_fucci_cytometry function
    plot_fucci_cytometry(fornma_file_path=fornma_file,
                         image_name_col=IMAGE_NAME_COL,
                         treatment_file=treatment_file,
                         output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
