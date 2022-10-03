# analyse annotators and models variability module

print('initializing...')  # noqa

# code destined to analysing annotators and
# models variability regarding cell count,
# cell position and cell shape.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from time import sleep
from os.path import join
from pandas import concat
from seaborn import boxplot
from pandas import DataFrame
from matplotlib import pyplot as plt
from src.utils.aux_funcs import get_data_from_consolidated_df
print('all required libraries successfully imported.')  # noqa
sleep(0.8)

######################################################################
# defining global parameters

# defining file/folder paths
CONSOLIDATED_DATAFRAME_PATH = join('.',
                                   'data',
                                   'consolidated_df',
                                   'consolidated_df.csv')
IMAGES_FOLDER_PATH = join('.',            # required in order to get image resolution
                          'data',         # (which is necessary to normalize centroids)
                          'main_folder',
                          'imgs',
                          'annotator_comparison')
OUTPUT_FOLDER_PATH = join('.',
                          'data',
                          'output',
                          'variability')
DETECTION_THRESHOLD = 0.5
ERROR_TYPE = 'counts'
ANNOTATORS = ['angelo',
              'camila',
              'debora',
              'fernanda',
              'samlai']

######################################################################
# defining auxiliary functions


def get_counts_error(detections_a: DataFrame,
                     detections_b: DataFrame
                     ) -> int:
    """
    Given two data frames representing cell
    detections, returns counts error between
    two groups.
    """
    # getting number of detections in df a
    detections_a_num = len(detections_a)

    # getting number of detections in df b
    detections_b_num = len(detections_b)

    # getting difference between number of detections
    detections_num_diff = detections_a_num - detections_b_num

    # getting absolute value for detections num difference
    detections_num_abs_diff = abs(detections_num_diff)

    # returning absolute difference
    return detections_num_abs_diff


def get_rmse(detections_a: DataFrame,
             detections_b: DataFrame
             ) -> float:
    """
    Given two data frames representing cell
    detections, returns root mean squared
    error between two groups' centroids.
    """
    pass


def get_shape_error(detections_a: DataFrame,
                    detections_b: DataFrame
                    ) -> float:
    """
    Given two data frames representing cell
    detections, returns root mean squared
    error between two groups' centroids.
    """
    pass


def get_single_image_annotators_error_df(image_name: str,
                                         image_group: DataFrame,
                                         error_type: str = 'counts'
                                         ) -> DataFrame:
    """
    Given annotators detections data frame,
    returns data frame of following structure:
    | image_name | annotator_a | annotator_b | {error_type}_error |
    |    img01   |    angelo   |   camila    |         5          |
    |    img01   |    angelo   |   debora    |         2          |
    ...
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # grouping df by evaluators
    evaluator_groups = image_group.groupby('evaluator')

    # iterating over evaluators
    for evaluator_name, evaluator_group in evaluator_groups:

        # filtering df for other evaluators detections
        other_evaluator_group = image_group.loc[image_group['evaluator'] != evaluator_name]

        # grouping other evaluators detections
        other_annotators_groups = other_evaluator_group.groupby('evaluator')

        # iterating over other annotators groups
        for other_annotator_name, other_annotator_group in other_annotators_groups:

            # defining placeholder value for current error
            error = None

            # getting error between two current annotators

            # checking desired error type

            # counts error
            if error_type == 'counts':

                # getting counts error
                error = get_counts_error(detections_a=evaluator_group,
                                         detections_b=other_annotator_group)

            # rmse
            elif error_type == 'rmse':

                # getting rmse
                error = get_rmse(detections_a=evaluator_group,
                                 detections_b=other_annotator_group)

            # shape error
            elif error_type == 'shape':

                # getting shape error
                error = get_shape_error(detections_a=evaluator_group,
                                        detections_b=other_annotator_group)

            # invalid input
            else:

                # printing error message
                e_string = f'Invalid input "{error_type}" to function "get_annotators_df"'
                print(e_string)
                exit()

            # defining current annotators error dict
            current_annotators_error_dict = {'image_name': image_name,
                                             'evaluator_a': evaluator_name,
                                             'evaluator_b': other_annotator_name,
                                             f'{error_type}_error': error}

            # defining current annotators error df
            current_annotators_error_df = DataFrame(current_annotators_error_dict,
                                                    index=[0])

            # appending current df to dfs list
            dfs_list.append(current_annotators_error_df)

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def get_multiple_images_evaluators_error_df(detections_df: DataFrame,
                                            error_type: str = 'counts'
                                            ) -> DataFrame:
    """
    Given annotators detections data frame,
    returns data frame of following structure:
    | image_name | annotator_a | annotator_b | {error_type}_error |
    |    img01   |    angelo   |   camila    |         5          |
    |    img01   |    angelo   |   debora    |         2          |
    ...
    |    img02   |    angelo   |   camila    |         4          |
    |    img02   |    angelo   |   debora    |         1          |
    ...
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # grouping annotators df by images
    image_groups = detections_df.groupby('file_name')

    # iterating over images
    for image_name, image_group in image_groups:

        # converting image name to string
        image_name_str = str(image_name)

        # getting current image error df
        current_image_error_df = get_single_image_annotators_error_df(image_name=image_name_str,
                                                                      image_group=image_group,
                                                                      error_type=error_type)

        # appending current image errors to dfs list
        dfs_list.append(current_image_error_df)

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def get_evaluators_pairs_means(errors_df: DataFrame,
                               annotators: list,
                               error_type: str = 'counts'
                               ) -> DataFrame:
    """
    Given an errors data frame, returns
    errors means/stds between evaluator
    pairs, for all images, in a data
    frame of following structure:
    | evaluator_a | evaluator_b | error_mean |
    |    angelo   |    camila   |     5.2    |
    |    angelo   |    debora   |     3.4    |
    ...
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # grouping df by evaluator
    evaluator_groups = errors_df.groupby('evaluator_a')

    # iterating over evaluator groups
    for evaluator_name, evaluator_group in evaluator_groups:

        # checking whether current evaluator is an annotator
        if evaluator_name in annotators:

            # filtering annotator group to get only other annotators
            other_annotators_df = evaluator_group[evaluator_group['evaluator_b'].isin(annotators)]

            # grouping other annotators by name
            other_annotators_groups = other_annotators_df.groupby('evaluator_b')

            # iterating over other annotators groups
            for other_annotator_name, other_annotator_group in other_annotators_groups:

                # getting current annotator pair errors
                current_annotator_pair_errors = other_annotator_group[f'{error_type}_error']

                # getting current annotator pair mean
                current_annotator_pair_mean = current_annotator_pair_errors.mean()

                # defining current annotator pair dict
                current_annotator_pair_dict = {'evaluator_a': evaluator_name,
                                               'evaluator_b': other_annotator_name,
                                               f'{error_type}_error_mean': current_annotator_pair_mean}

                # defining current annotator pair df
                current_annotator_pair_df = DataFrame(current_annotator_pair_dict,
                                                      index=[0])

                # appending current annotator pair df to dfs list
                dfs_list.append(current_annotator_pair_df)

        # if evaluator is ml model
        else:

            # filtering annotator group to get only annotators
            annotators_df = evaluator_group[evaluator_group['evaluator_b'].isin(annotators)]

            # grouping annotators by name
            annotators_groups = annotators_df.groupby('evaluator_b')

            # iterating over annotators groups
            for annotator_name, annotator_group in annotators_groups:

                # getting current evaluator pair errors
                current_evaluator_pair_errors = annotator_group[f'{error_type}_error']

                # getting current evaluator pair mean
                current_evaluator_pair_mean = current_evaluator_pair_errors.mean()

                # defining current evaluator pair dict
                current_evaluator_pair_dict = {'evaluator_a': evaluator_name,
                                               'evaluator_b': annotator_name,
                                               f'{error_type}_error_mean': current_evaluator_pair_mean}

                # defining current evaluator pair df
                current_evaluator_pair_df = DataFrame(current_evaluator_pair_dict,
                                                      index=[0])

                # appending current evaluator pair df to dfs list
                dfs_list.append(current_evaluator_pair_df)

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def get_evaluator_means_stds(evaluator_errors_df: DataFrame,
                             error_type: str = 'counts'
                             ) -> DataFrame:
    """
    Given an evaluator errors data frame,
    returns mean/stds values, in a data
    frame of following structure:
    | evaluator | error_mean | error_std |
    |   angelo  |     5.3    |    1.2    |
    |   camila  |     4.8    |    0.9    |
    | retinanet |     7.9    |    2.9    |
    ...
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # grouping df by evaluator
    evaluator_groups = evaluator_errors_df.groupby('evaluator_a')

    # iterating over evaluator groups
    for evaluator_name, evaluator_group in evaluator_groups:

        # getting errors column
        current_evaluator_errors = evaluator_group[f'{error_type}_error_mean']

        # getting current evaluator mean/std
        current_evaluator_mean = current_evaluator_errors.mean()
        current_evaluator_std = current_evaluator_errors.std()

        # defining current evaluator dict
        current_evaluator_dict = {'evaluator': evaluator_name,
                                  'error_mean': current_evaluator_mean,
                                  'error_std': current_evaluator_std}

        # defining current evaluator df
        current_evaluator_df = DataFrame(current_evaluator_dict,
                                         index=[0])

        # appending current evaluator df to dfs list
        dfs_list.append(current_evaluator_df)

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def plot_box_plots(evaluator_errors_df: DataFrame,
                   error_type: str = 'counts'
                   ) -> None:
    """
    Given bell curves data frame, plots
    multiple box plots in same figure.
    """
    # plotting curves
    boxplot(data=evaluator_errors_df,
            x='evaluator_a',
            y=f'{error_type}_error_mean')

    # setting title/labels
    error_type = error_type.capitalize()
    plt.title(f'{error_type} Errors Distributions by Evaluator')
    plt.xlabel('Evaluator')
    plt.ylabel(f'{error_type} Error')

    # showing plot
    plt.show()


def main():
    """
    Runs main code.
    """
    # getting data from consolidated df csv
    print('getting data from consolidated df...')
    main_df = get_data_from_consolidated_df(consolidated_df_file_path=CONSOLIDATED_DATAFRAME_PATH)

    # filtering data frame by annotator comparison images
    print('filtering df by annotator comparison...')
    filtered_dataframe = main_df.loc[main_df['type'] == 'annotator_comparison']

    # filtering data frame by detection threshold
    print('filtering df by detection threshold...')
    filtered_dataframe = filtered_dataframe.loc[filtered_dataframe['detection_threshold'] >= DETECTION_THRESHOLD]

    # getting errors data frame
    print(f'getting {ERROR_TYPE} errors data frame...')
    total_errors_df = get_multiple_images_evaluators_error_df(detections_df=filtered_dataframe,
                                                              error_type=ERROR_TYPE)

    # getting evaluator pair mean
    print(f'getting {ERROR_TYPE} evaluator errors data frame...')
    evaluator_errors_df = get_evaluators_pairs_means(errors_df=total_errors_df,
                                                     annotators=ANNOTATORS,
                                                     error_type=ERROR_TYPE)

    # getting evaluator means/stds
    evaluator_means_df = get_evaluator_means_stds(evaluator_errors_df=evaluator_errors_df,
                                                  error_type=ERROR_TYPE)

    # defining plot/statistics output name
    stats_output_name = 'statistics.xlsx'

    # defining output paths
    stats_output_path = join(OUTPUT_FOLDER_PATH,
                             stats_output_name)

    # saving statistics
    # TODO: add statistics calculations here

    # saving errors box plots
    print('plotting errors variability box plots...')
    plot_box_plots(evaluator_errors_df=evaluator_errors_df,
                   error_type=ERROR_TYPE)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
