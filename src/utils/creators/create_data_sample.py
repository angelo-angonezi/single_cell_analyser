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


def add_dataset_col(df: DataFrame,
                    train_size: float,
                    val_size: float,
                    test_size: float
                    ) -> None:
    """
    Given an annotations df, groups
    df by grouper cols, and adds
    dataset column, balancing train/test
    according to the df groups.
    """
    # defining split col name
    split_col_name = 'split'

    # adding placeholder "split" col
    df[split_col_name] = None

    # defining group cols
    group_cols = ['treatment',
                  'class_group']

    # grouping df
    df_groups = df.groupby(group_cols)

    # getting groups/rows num
    groups_num = len(df_groups)
    rows_num = len(df)

    # printing execution message
    f_string = f'{groups_num} groups were found based on: {group_cols}'
    print(f_string)

    # defining starter for indices
    current_group_index = 1
    current_row_index = 1

    # iterating over groups
    for df_name, df_group in df_groups:

        # defining execution message
        base_string = f'adding data split col to group '
        base_string += f'{current_group_index} of {groups_num} '
        base_string += f'| row: #INDEX# of #TOTAL#'

        # randomly splitting current group rows
        current_test_split = df_group.sample(frac=test_size,
                                             random_state=SEED)
        rest_df = df_group.drop(current_test_split.index)
        val_frac = val_size / (train_size + val_size)
        current_val_split = rest_df.sample(frac=val_frac,
                                           random_state=SEED)
        current_train_split = rest_df.drop(current_val_split.index)

        # getting train/test indices
        train_indices = current_train_split.index
        val_indices = current_val_split.index
        test_indices = current_test_split.index

        # adding split column based on current samples

        # iterating over train indices
        for train_index in train_indices:

            # printing execution message
            print_progress_message(base_string=base_string,
                                   index=current_row_index,
                                   total=rows_num)

            # updating current row split col
            df.at[train_index, split_col_name] = 'train'

            # updating current row index
            current_row_index += 1

        # iterating over val indices
        for val_index in val_indices:

            # printing execution message
            print_progress_message(base_string=base_string,
                                   index=current_row_index,
                                   total=rows_num)

            # updating current row split col
            df.at[val_index, split_col_name] = 'val'

            # updating current row index
            current_row_index += 1

        # iterating over test indices
        for test_index in test_indices:

            # printing execution message
            print_progress_message(base_string=base_string,
                                   index=current_row_index,
                                   total=rows_num)

            # updating current row split col
            df.at[test_index, split_col_name] = 'test'

            # updating current row index
            current_row_index += 1

        # updating current group index
        current_group_index += 1


def add_class_group_col(df: DataFrame) -> None:
    """
    Given an annotated crops info df,
    adds class group col, adapting it
    should it be a number.
    """
    # getting first element info
    first_row = df.iloc[0]
    first_class = first_row['class']
    first_class_is_num = isinstance(first_class, float)

    # checking whether class is a number
    if first_class_is_num:

        # adding class group col
        df['class_group'] = df['class'].round()
        df['class_group'] = df['class_group'].astype(int)
        df['class_group'] = df['class_group'].astype(str)

    else:

        # adding class group col
        df['class_group'] = df['class'].astype(str)


def get_sample_df(df: DataFrame,
                  sample_size: int
                  ) -> DataFrame:
    """
    Given a crops info / annotations df,
    randomly (but balanced) selects examples
    from each class, returning a samples df.
    """
    # defining group col
    group_col = 'class_group'

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
    cols_to_keep = ['crop_name',
                    'treatment',
                    'class']
    crops_df = crops_df[cols_to_keep]

    # adding class_group col
    print('adding class group col...')
    add_class_group_col(df=crops_df)

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
