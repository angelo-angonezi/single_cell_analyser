# single cell analyser module

print('initializing...')  # noqa

# code destined to automating single cell
# feature extraction and analyses.

######################################################################
# importing required libraries

print('importing required libraries...')  # noqa
from time import sleep
from os.path import join
from src.utils.aux_funcs import get_data_from_consolidated_df
print('all required libraries successfully imported.')  # noqa
sleep(0.8)

######################################################################
# defining auxiliary functions

######################################################################
# defining main function

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
