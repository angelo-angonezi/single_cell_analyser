# create data splits module

print('initializing...')  # noqa

# Code destined to splitting data for
# single cell crops regression and/or
# classification networks.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
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

TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.1
TEST_SPLIT = 0.3
SEED = 53

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'create data splits module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # crops info file param
    parser.add_argument('-c', '--crops-info-file',
                        dest='crops_info_file',
                        required=True,
                        help='defines path to annotated crops info df (.csv) file')

    # images folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing source images.')

    # images extension param
    parser.add_argument('-e', '--extension',
                        dest='extension',
                        required=True,
                        help='defines images extension (.png, .jpg, .tif).')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to folder containing train/val/test subfolders.')

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

    # defining group col
    group_col = 'class_group'

    # grouping df
    df_groups = df.groupby(group_col)

    # getting groups/rows num
    groups_num = len(df_groups)
    rows_num = len(df)

    # printing execution message
    f_string = f'{groups_num} groups were found based on: {group_col}'
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
    Given a crops info df,
    adds class group col,
    adapting it should it be
    a number.
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


def create_data_splits(crops_info_file: str,
                       images_folder: str,
                       extension: str,
                       output_folder: str
                       ) -> None:
    # reading crops info file
    print('reading crops info file...')
    crops_df = read_csv(crops_info_file)

    # dropping unrequired columns
    print('dropping unrequired columns...')
    cols_to_keep = ['crop_name',
                    'class']
    crops_df = crops_df[cols_to_keep]

    # adding class_group col
    print('adding class group col...')
    add_class_group_col(df=crops_df)

    # adding dataset col
    print('adding dataset col...')
    add_dataset_col(df=crops_df,
                    train_size=TRAIN_SPLIT,
                    val_size=VAL_SPLIT,
                    test_size=TEST_SPLIT)

    # saving df
    print('saving dataset df...')
    save_name = 'dataset_df.csv'
    save_path = join(output_folder,
                     save_name)
    crops_df.to_csv(save_path,
                    index=False)

    # getting images num
    images_num = len(crops_df)

    # printing execution message
    print('getting images...')

    # getting splits dfs
    train_df = crops_df[crops_df['split'] == 'train']
    val_df = crops_df[crops_df['split'] == 'val']
    test_df = crops_df[crops_df['split'] == 'test']

    # getting splits paths
    train_files = train_df['crop_name'].to_list()
    val_files = val_df['crop_name'].to_list()
    test_files = test_df['crop_name'].to_list()

    # getting split sizes
    train_size = len(train_files)
    val_size = len(val_files)
    test_size = len(test_files)

    # getting split ratios
    train_ratio = train_size / images_num
    val_ratio = val_size / images_num
    test_ratio = test_size / images_num

    # getting split percentages
    train_percentage = train_ratio * 100
    val_percentage = val_ratio * 100
    test_percentage = test_ratio * 100

    # rounding values
    train_percentage_round = round(train_percentage)
    val_percentage_round = round(val_percentage)
    test_percentage_round = round(test_percentage)

    # getting subfolder paths
    train_folder = join(output_folder, 'train')
    val_folder = join(output_folder, 'val')
    test_folder = join(output_folder, 'test')

    # printing execution message
    f_string = f'found {images_num} (100%) images in input folder.\n'
    f_string += f'{train_size} ({train_percentage_round}%) will be copied to "train" folder.\n'
    f_string += f'{val_size} ({val_percentage_round}%) will be copied to "val" folder.\n'
    f_string += f'{test_size} ({test_percentage_round}%) will be copied to "test" folder.'
    print(f_string)

    # copying train images
    print('copying train images...')
    copy_multiple_files(src_folder_path=images_folder,
                        dst_folder_path=train_folder,
                        files_list=train_files,
                        file_extension=extension)

    # copying val images
    print('copying val images...')
    copy_multiple_files(src_folder_path=images_folder,
                        dst_folder_path=val_folder,
                        files_list=val_files,
                        file_extension=extension)

    # copying test images
    print('copying test images...')
    copy_multiple_files(src_folder_path=images_folder,
                        dst_folder_path=test_folder,
                        files_list=test_files,
                        file_extension=extension)

    # printing execution message
    print('data splits created!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting crops info file param
    crops_info_file = args_dict['crops_info_file']

    # getting images folder param
    images_folder = args_dict['images_folder']

    # getting images extension param
    extension = args_dict['extension']

    # getting output folder param
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # splitting data
    create_data_splits(crops_info_file=crops_info_file,
                       images_folder=images_folder,
                       extension=extension,
                       output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
