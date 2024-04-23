# create data sample module

print('initializing...')  # noqa

# Code destined to sampling data for
# single cell crops regression and/or
# classification networks.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import copy_multiple_files
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

SAMPLE_SIZE = 100
SEED = 53

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'create data sample module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # annotations file param
    parser.add_argument('-a', '--annotations-file',
                        dest='annotations_file',
                        required=True,
                        help='defines path to ANNOTATED crops info df (.csv) file')

    # images folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing crops.')

    # images extension param
    parser.add_argument('-x', '--extension',
                        dest='extension',
                        required=True,
                        help='defines images extension (.png, .jpg, .tif).')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to folder which will contain image sample.')

    # sample size param
    parser.add_argument('-s', '--sample-size',
                        dest='sample_size',
                        required=False,
                        default=SAMPLE_SIZE,
                        help='defines number of images to be randomly (but balanced) selected for sample.')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_sample_df(df: DataFrame,
                  sample_size: int
                  ) -> DataFrame:
    """
    Given a crops info / annotations df,
    randomly (but balanced) selects examples
    from each class, returning a samples df.
    """
    # defining group col
    group_col = 'class'

    # checking class group lengths
    df_groups = df.groupby(group_col)
    groups_num = len(df_groups)

    # defining placeholder value for dfs list
    dfs_list = []

    # getting corrected sample size for each group
    group_size = int(sample_size / groups_num)

    # defining starter for current_group_index
    current_group_index = 1

    # iterating over df groups
    for df_name, df_group in df_groups:

        # printing execution message
        base_string = f'getting sample for group "{df_name}" | #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_group_index,
                               total=groups_num)

        # getting df group sample (applying undersampling)
        current_sample = df_group.sample(n=group_size,
                                         random_state=SEED)

        # appending current sample to dfs list
        dfs_list.append(current_sample)

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning balanced df
    return final_df


def create_data_sample(annotations_file: str,
                       images_folder: str,
                       extension: str,
                       output_folder: str,
                       sample_size: int
                       ) -> None:
    # reading annotations file
    print('reading annotations file...')
    crops_df = read_csv(annotations_file)

    # dropping unrequired columns
    print('dropping unrequired columns...')
    cols_to_keep = ['crop_name', 'class']
    crops_df = crops_df[cols_to_keep]

    # getting sample df
    print('getting sample df...')
    crops_df = get_sample_df(df=crops_df,
                             sample_size=sample_size)

    # saving df
    print('saving sample df...')
    save_name = 'sample_df.csv'
    save_path = join(output_folder,
                     save_name)
    crops_df.to_csv(save_path,
                    index=False)

    # getting sample files list
    crop_name_col = crops_df['crop_name']
    sample_files = crop_name_col.to_list()

    # copying sample images
    print('copying sample images...')
    copy_multiple_files(src_folder_path=images_folder,
                        dst_folder_path=output_folder,
                        files_list=sample_files,
                        file_extension=extension)

    # printing execution message
    print('data sample created!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting annotations file param
    annotations_file = args_dict['annotations_file']

    # getting images folder param
    images_folder = args_dict['images_folder']

    # getting images extension param
    extension = args_dict['extension']

    # getting output folder param
    output_folder = args_dict['output_folder']

    # getting sample size param
    sample_size = int(args_dict['sample_size'])

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # splitting data
    create_data_sample(annotations_file=annotations_file,
                       images_folder=images_folder,
                       extension=extension,
                       output_folder=output_folder,
                       sample_size=sample_size)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
