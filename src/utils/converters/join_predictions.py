# join detection results module

print('initializing...')  # noqa

# Given a path to a folder containing model detection files
# (multiple det_*class_name*.txt) joins them into a single file.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import merge
from pandas import concat
from os.path import exists
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from pandas.errors import EmptyDataError
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
    description = "join detection results"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # crops info file param
    parser.add_argument('-c', '--crops-info-file',
                        dest='crops_info_file',
                        required=True,
                        help='defines path to annotated crops info df (.csv) file')

    # autophagy predictions file param
    parser.add_argument('-a', '--autophagy-predictions-file',
                        dest='autophagy_predictions_file',
                        required=False,
                        help='defines path to autophagy predictions df (.csv) file')

    # dna_damage predictions file param
    parser.add_argument('-d', '--dna-damage-predictions-file',
                        dest='dna_damage_predictions_file',
                        required=False,
                        help='defines path to dna damage predictions df (.csv) file')

    # nii predictions file param
    parser.add_argument('-n', '--nii-predictions-file',
                        dest='nii_predictions_file',
                        required=False,
                        help='defines path to nii predictions df (.csv) file')

    # cell cycle predictions file param
    parser.add_argument('-cc', '--cell-cycle-predictions-file',
                        dest='cell_cycle_predictions_file',
                        required=False,
                        help='defines path to cell cycle predictions df (.csv) file')

    # erk predictions file param
    parser.add_argument('-e', '--erk-predictions-file',
                        dest='erk_predictions_file',
                        required=False,
                        help='defines path to erk predictions df (.csv) file')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        type=str,
                        help='defines output path (.csv)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def merge_predictions_df(base_df: DataFrame,
                         predictions_file: str,
                         phenotype: str
                         ) -> DataFrame:
    """
    Given a base dataframe, and a path
    to a file containing predictions,
    merges dfs based on crop_name col,
    adding new column based on given
    phenotype.
    """
    # checking whether given predictions file is none
    if predictions_file is None:

        # return base df unaltered
        return base_df

    # checking whether given predictions file exists
    predictions_file_exists = exists(predictions_file)

    # if it does not exist
    if not predictions_file_exists:

        # return base df unaltered
        return base_df

    # if it does exist, read predictions file
    predictions_df = read_csv(predictions_file)

    # renaming prediction column to given phenotype
    new_cols = ['crop_name', phenotype]
    predictions_df.columns = new_cols

    # merging dfs
    merged_df = merge(left=base_df,
                      right=predictions_df,
                      on='crop_name')

    # returning merged df
    return merged_df


def join_predictions_df(crops_info_file: str,
                        autophagy_predictions_file,
                        dna_damage_predictions_file,
                        nii_predictions_file,
                        cell_cycle_predictions_file,
                        erk_predictions_file
                        ) -> DataFrame:
    """
    Given a path to a crops info df, and
    corresponding prediction dfs, merges
    dfs based on crop_name column, returning
    joined data frame.
    """
    # reading crops info file
    print('reading crops info file...')
    crops_info_df = read_csv(crops_info_file)

    # merging prediction dfs
    print('merging prediction dfs...')
    predictions_df = merge_predictions_df(base_df=crops_info_df,
                                          predictions_file=autophagy_predictions_file,
                                          phenotype='autophagy')
    predictions_df = merge_predictions_df(base_df=predictions_df,
                                          predictions_file=dna_damage_predictions_file,
                                          phenotype='dna_damage')
    predictions_df = merge_predictions_df(base_df=predictions_df,
                                          predictions_file=nii_predictions_file,
                                          phenotype='nii')
    predictions_df = merge_predictions_df(base_df=predictions_df,
                                          predictions_file=cell_cycle_predictions_file,
                                          phenotype='cell_cycle')
    predictions_df = merge_predictions_df(base_df=predictions_df,
                                          predictions_file=erk_predictions_file,
                                          phenotype='erk')

    # returning predictions df
    return predictions_df


def save_joined_dataframe(joined_df: DataFrame,
                          output_path: str
                          ) -> None:
    """
    Given a joined predictions data frame, and a
    path to an output csv file, saves df in given
    output.
    """
    # printing execution message
    print('saving dataframe...')

    # saving dataframe to output path
    joined_df.to_csv(output_path,
                     index=False)

    # printing execution message
    print(f'joined dataframe saved at "{output_path}"')

######################################################################
# defining main function


def main():
    """
    Gets arguments from cli and runs main code.
    """
    # getting args dict
    args_dict = get_args_dict()

    # getting crops info file param
    crops_info_file = args_dict['crops_info_file']

    # getting predictions files params
    autophagy_predictions_file = args_dict['autophagy_predictions_file']
    dna_damage_predictions_file = args_dict['dna_damage_predictions_file']
    nii_predictions_file = args_dict['nii_predictions_file']
    cell_cycle_predictions_file = args_dict['cell_cycle_predictions_file']
    erk_predictions_file = args_dict['erk_predictions_file']

    # getting output path
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # creating joined data frame
    joined_df = join_predictions_df(crops_info_file=crops_info_file,
                                    autophagy_predictions_file=autophagy_predictions_file,
                                    dna_damage_predictions_file=dna_damage_predictions_file,
                                    nii_predictions_file=nii_predictions_file,
                                    cell_cycle_predictions_file=cell_cycle_predictions_file,
                                    erk_predictions_file=erk_predictions_file)

    # saving joined data frame
    save_joined_dataframe(joined_df=joined_df,
                          output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
