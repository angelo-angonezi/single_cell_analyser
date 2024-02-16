# NIIRegression augment data module

print('initializing...')  # noqa

# Code destined to augmenting data for
# NIIRegression classification network.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import augment_image
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'NIIRegression data augmentation module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # dataset file param
    parser.add_argument('-d', '--dataset-file',
                        dest='dataset_file',
                        required=True,
                        help='defines path to dataset df (.csv) file')

    # input folder param
    parser.add_argument('-i', '--splits-folder',
                        dest='splits_folder',
                        required=True,
                        help='defines path to folder containing train/val subfolders.')

    # images extension param
    parser.add_argument('-e', '--extension',
                        dest='extension',
                        required=True,
                        help='defines images extension (.png, .jpg, .tif).')

    # output folder param
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help='defines path to folder which will contain augmented images (must contain train/val subfolders).')  # noqa

    # resize param
    parser.add_argument('-r', '--resize',
                        dest='resize',
                        action='store_true',
                        required=False,
                        default=False,
                        help='defines whether or not to resize images.')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def augment_data_split(df: DataFrame,
                       split: str,
                       splits_folder: str,
                       extension: str,
                       output_folder: str,
                       resize: bool
                       ) -> None:
    """
    Given a dataset df and a folder
    containing respective images,
    augments data specified by
    given split parameter, saving
    augmented data to output folder.
    """
    # filtering df by split
    split_df = df[df['split'] == split]

    # getting df rows
    df_rows = split_df.iterrows()

    # updating output folder
    output_folder = join(output_folder,
                         split)

    # getting rows num
    rows_num = len(split_df)

    # defining starter for current row index
    current_row_index = 1

    # iterating over df rows
    for row_index, row_data in df_rows:

        # printing execution message
        base_string = f'augmenting data for "{split}" split #INDEX# #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row file path
        file_name = row_data['crop_name']

        # augmenting current image
        augment_image(image_name=file_name,
                      extension=extension,
                      images_folder=splits_folder,
                      output_folder=output_folder,
                      resize=resize)

        # updating current row index
        current_row_index += 1


def get_augmented_df(df: DataFrame) -> DataFrame:
    """
    Given a dataset df, adds
    augmented images rows,
    returning updated df.
    """
    # defining placeholder value for dfs list
    dfs_list = []

    # getting df rows
    df_rows = df.iterrows()

    # getting rows num
    rows_num = len(df)

    # defining augmentation "extensions"
    augmentation_list = ['o',
                         'r',
                         'v',
                         'h',
                         'od',
                         'rd',
                         'vd',
                         'hd',
                         'ou',
                         'ru',
                         'vu',
                         'hu']

    # defining starter for current row index
    current_row_index = 1

    # iterating over df rows
    for row_index, row_data in df_rows:

        # printing execution message
        base_string = 'getting augmented df for image #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row file name
        current_name = row_data['crop_name']

        # getting current row file class
        current_class = row_data['class']

        # getting current row file split
        current_split = row_data['split']

        # getting current file expanded list
        expanded_names = [f'{current_name}_{expansion}'
                          for expansion
                          in augmentation_list]

        # getting expanded class list
        expanded_class = [current_class for _ in expanded_names]

        # getting expanded split list
        expanded_splits = [current_split for _ in expanded_names]

        # assembling current image dict
        current_dict = {'crop_name': expanded_names,
                        'class': expanded_class,
                        'split': expanded_splits}

        # assembling current image df
        current_df = DataFrame(current_dict)

        # appending current df to dfs list
        dfs_list.append(current_df)

        # updating current row index
        current_row_index += 1

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning updated df
    return final_df


def augment_data(dataset_file: str,
                 splits_folder: str,
                 extension: str,
                 output_folder: str,
                 resize: bool
                 ) -> None:
    # reading dataset df
    print('reading dataset df...')
    dataset_df = read_csv(dataset_file)

    # augmenting train data
    print('augmenting train data...')
    augment_data_split(df=dataset_df,
                       split='train',
                       splits_folder=splits_folder,
                       extension=extension,
                       output_folder=output_folder,
                       resize=resize)

    # augmenting val data
    print('augmenting val data...')
    augment_data_split(df=dataset_df,
                       split='val',
                       splits_folder=splits_folder,
                       extension=extension,
                       output_folder=output_folder,
                       resize=resize)

    # getting augmented df
    print('getting augmented df..')
    augmented_df = get_augmented_df(df=dataset_df)
    augmented_num = len(augmented_df)

    # saving augmented images dataset df
    print('saving augmented images dataset df...')
    save_name = 'augmented_dataset_df.csv'
    save_path = join(output_folder,
                     save_name)
    augmented_df.to_csv(save_path,
                        index=False)

    # printing execution message
    print('augmentation complete!')
    print(f'a total of {augmented_num} images have been saved to augmentation folder')
    print(f'results saved to "{output_folder}".')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting dataset file param
    dataset_file = args_dict['dataset_file']

    # getting splits folder param
    splits_folder = args_dict['splits_folder']

    # getting images extension param
    extension = args_dict['extension']

    # getting output folder param
    output_folder = args_dict['output_folder']

    # getting resize param
    resize = bool(args_dict['resize'])

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running augment_data function
    augment_data(dataset_file=dataset_file,
                 splits_folder=splits_folder,
                 extension=extension,
                 output_folder=output_folder,
                 resize=resize)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
