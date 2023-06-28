# plot histograms (area/axis_ratio) module

print('initializing...')  # noqa

# Code destined to plotting area and axis ratio
# histograms for model and annotator data.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from os.path import exists
from pandas import read_csv
from pandas import DataFrame
from seaborn import histplot
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import add_nma_col
from src.utils.aux_funcs import add_date_col
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import add_treatment_col_fer
from src.utils.aux_funcs import add_treatment_col_daph
from src.utils.aux_funcs import add_treatment_col_debs
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_merged_detection_annotation_df
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

DETECTION_THRESHOLD = 0.5

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "plot area histograms module"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # detection file param
    detection_help = 'defines path to csv file containing model detections'
    parser.add_argument('-d', '--detection_file',
                        dest='detection_file',
                        required=True,
                        help=detection_help)

    # gt file param
    gt_help = 'defines path to csv file containing ground-truth annotations in model format'
    parser.add_argument('-g', '--ground-truth-file',
                        dest='ground_truth_file',
                        required=True,
                        help=gt_help)

    # fornma file param
    fornma_help = 'defines path to csv file containing ground-truth annotations in fornma format'
    parser.add_argument('-f', '--fornma-file',
                        dest='fornma_file',
                        required=True,
                        help=fornma_help)

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


def get_nma_df(df: DataFrame,
               tt: str
               ) -> DataFrame:
    """
    Given a merged detections/annotations data frame,
    returns a new df, of following structure:
    | evaluator | cell_area | axis_ratio |
    |   model   |   633.6   |   1.60913  |
    |   fornma  |   267.1   |   1.77106  |
    ...
    :param df: DataFrame. Represents merged detections/annotations data.
    :param tt: String. Represents treatment type (researcher name).
    :return: None.
    """
    # adding date column to df
    print('adding date column to df...')
    add_date_col(df=df)

    # adding area column to df
    print('adding area column to df...')
    add_nma_col(df=df,
                col_name='area')

    # adding axis ratio column to df
    print('adding axis ratio column to df...')
    add_nma_col(df=df,
                col_name='axis_ratio')

    # adding treatment column to df
    print('adding treatment column to df...')
    if tt == 'daph':
        add_treatment_col_daph(df=df,
                               data_format='model')
    else:
        print('treatment type undefined.')
        exit()

    # dropping unrequired cols
    all_cols = df.columns.to_list()
    keep_cols = ['area', 'axis_ratio', 'evaluator', 'img_file_name', 'treatment', 'date']
    drop_cols = [col
                 for col
                 in all_cols
                 if col not in keep_cols]
    final_df = df.drop(drop_cols,
                       axis=1)

    # returning final df
    return final_df


def get_fornma_df(fornma_file_path: str,
                  tt: str
                  ) -> DataFrame:
    """
    Given a merged detections/annotations data frame,
    returns a new df, of following structure:
    | evaluator | cell_area | axis_ratio |
    |   model   |   633.6   |   1.60913  |
    |   fornma  |   267.1   |   1.77106  |
    ...
    :param fornma_file_path: String. Represents a path to a file.
    :param tt: String. Represents treatment type (researcher name).
    :return: None.
    """
    # reading input df
    df = read_csv(fornma_file_path)

    # adding treatment column to df
    print('adding treatment column to df...')
    if tt == 'daph':
        add_treatment_col_daph(df=df,
                               data_format='fornma')
    else:
        print('treatment type undefined.')
        exit()

    # dropping unrequired cols
    all_cols = df.columns.to_list()
    keep_cols = ['Area', 'Radius_ratio', 'Image_name_red', 'treatment']
    drop_cols = [col
                 for col
                 in all_cols
                 if col not in keep_cols]
    final_df = df.drop(drop_cols,
                       axis=1)

    # renaming cols
    final_df.columns = ['area', 'axis_ratio', 'img_file_name', 'treatment']

    # removing file extension from name column
    current_names = final_df['img_file_name']
    new_names = [f.replace('.tif', '') for f in current_names]
    final_df['img_file_name'] = new_names

    # adding date column to df
    print('adding date column to df...')
    add_date_col(df=final_df)

    # adding evaluator column
    final_df['evaluator'] = 'fornma_nuc'

    # returning final df
    return final_df


def generate_histograms(df: DataFrame,
                        output_folder: str,
                        col_name: str
                        ) -> None:
    """
    Given a nma data frame, plots area or axis_ratio
    histograms, filtering df by control group.
    :param df: DataFrame. Represents NMA data.
    :param output_folder: String. Represents a path to a folder.
    :param col_name: String. Represents a column name ('area' or 'axis_ratio').
    :return: None.
    """
    # grouping df
    df_groups = df.groupby(['evaluator', 'date'])

    # iterating over df_groups
    for df_name, df_group in df_groups:

        # getting evaluator/date
        evaluator, date = df_name

        # printing execution message
        f_string = f'plotting {col_name} histogram ({evaluator}, {date} group)...'
        print(f_string)

        # setting figure size
        plt.figure(figsize=(20, 10))

        # plotting data
        histplot(data=df_group,
                 x=col_name,
                 hue='treatment',
                 kde=True,
                 stat='count')

        # TODO: check w Guido changes to previous line based on documentation below.
        """
        'stat' param
        Aggregate statistic to compute in each bin.
            count: show the number of observations in each bin
            frequency: show the number of observations divided by the bin width
            probability or proportion: normalize such that bar heights sum to 1
            percent: normalize such that bar heights sum to 100
            density: normalize such that the total area of the histogram equals 1
        from: https://seaborn.pydata.org/generated/seaborn.histplot.html
        """

        # setting plot title
        plt_title = f'{evaluator} {col_name} histogram (date: {date})'
        plt.title(plt_title)

        # saving plot
        save_name = f'{col_name}_histograms_{evaluator}_{date}.png'
        save_path = join(output_folder,
                         save_name)
        plt.savefig(save_path)

        # closing plot
        plt.close()


def get_histograms_df(detection_file_path: str,
                      ground_truth_file_path: str,
                      fornma_file_path: str,
                      output_folder: str
                      ) -> DataFrame:
    """
    Given paths to model detections and gt annotations,
    returns merged dfs to plot area/axis_ratio histograms.
    :param detection_file_path: String. Represents a file path.
    :param ground_truth_file_path: String. Represents a file path.
    :param fornma_file_path: String. Represents a file path.
    :param output_folder: String. Represents a folder path.
    :return: None.
    """
    # defining csv output path
    save_name = 'histograms_df.csv'
    save_path = join(output_folder,
                     save_name)

    # defining placeholder value for final_df
    final_df = None

    # checking if csv output already exists
    if exists(save_path):

        # reading already existent data frame
        print('reading already existent data frame...')
        final_df = read_csv(save_path)

    # if output csv does not already exist
    else:

        # getting merged detections df
        print('getting data from input files...')
        merged_df = get_merged_detection_annotation_df(detections_df_path=detection_file_path,
                                                       annotations_df_path=ground_truth_file_path)

        # filtering df by detection threshold
        print('filtering df by detection threshold...')
        filtered_df = merged_df[merged_df['detection_threshold'] >= DETECTION_THRESHOLD]

        # getting nma df
        nma_df = get_nma_df(df=filtered_df,
                            tt='daph')

        # getting fornma df
        fornma_df = get_fornma_df(fornma_file_path=fornma_file_path,
                                  tt='daph')

        # adding date col to fornma df
        add_date_col(df=fornma_df)

        # concatenating nma/fornma dfs
        dfs_list = [nma_df, fornma_df]
        final_df = concat(dfs_list)

    # saving output csv
    final_df.to_csv(save_path,
                    index=False)

    # returning final_df
    return final_df


def plot_histograms(detection_file_path: str,
                    ground_truth_file_path: str,
                    fornma_file_path: str,
                    output_folder: str
                    ) -> None:
    """
    Given paths to model detections and gt annotations,
    merged dfs and plots area/axis_ratio histograms.
    :param detection_file_path: String. Represents a file path.
    :param ground_truth_file_path: String. Represents a file path.
    :param fornma_file_path: String. Represents a file path.
    :param output_folder: String. Represents a folder path.
    :return: None.
    """
    # getting histograms df
    spacer()
    print('getting histograms df...')
    histograms_df = get_histograms_df(detection_file_path=detection_file_path,
                                      ground_truth_file_path=ground_truth_file_path,
                                      fornma_file_path=fornma_file_path,
                                      output_folder=output_folder)

    # plotting histograms
    spacer()
    print('plotting area histograms...')
    generate_histograms(df=histograms_df,
                        output_folder=output_folder,
                        col_name='area')

    spacer()
    print('plotting axis ratio histograms...')
    generate_histograms(df=histograms_df,
                        output_folder=output_folder,
                        col_name='axis_ratio')

    # printing execution message
    spacer()
    print(f'results saved in folder "{output_folder}"')
    print('analysis complete.')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting detection file path
    detection_file = args_dict['detection_file']

    # getting ground-truth file path
    ground_truth_file = args_dict['ground_truth_file']

    # getting fornma file path
    fornma_file = args_dict['fornma_file']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running plot_area_histograms function
    plot_histograms(detection_file_path=detection_file,
                    ground_truth_file_path=ground_truth_file,
                    fornma_file_path=fornma_file,
                    output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
