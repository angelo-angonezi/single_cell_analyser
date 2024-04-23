# generate manual annotations module

print('initializing...')  # noqa

# Code destined to generating
# manual annotations file.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import get_dirs_in_folder
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
    description = 'generate manual annotations module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # images folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing classes subfolders (filled with respective images).')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help='defines extension (.tif, .png, .jpg) of images in input folders')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output file (.csv)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def generate_manual_annotations(images_folder: str,
                                images_extension: str,
                                output_path: str
                                ) -> None:
    """
    Given a path to a folder containing
    subfolders for each class, each
    containing respective images,
    creates a data frame of following
    structure:
    | crop_name | class |
    |  img001   |   G1  |
    ...
    """
    # getting subfolders in input folder
    print('getting subfolders in input folder...')
    subfolders = get_dirs_in_folder(path_to_folder=images_folder)

    # defining placeholder value for dfs list
    dfs_list = []

    # iterating over subfolders
    for subfolder in subfolders:

        # printing execution message
        f_string = f'getting manual annotations df for subfolder/class "{subfolder}"'
        print(f_string)

        # getting current subfolder path
        subfolder_path = join(images_folder,
                              subfolder)

        # getting files in current subfolder
        files_in_subfolder = get_specific_files_in_folder(path_to_folder=subfolder_path,
                                                          extension=images_extension)

        # assembling class col based on subfolder name
        class_col = [subfolder for _ in files_in_subfolder]

        # assembling current class dict
        current_dict = {'crop_name': files_in_subfolder,
                        'class': class_col}

        # assembling current class df
        current_df = DataFrame(current_dict)

        # appending current df to dfs list
        dfs_list.append(current_df)

    # concatenating dfs in dfs list
    crops_df = concat(dfs_list,
                      ignore_index=True)

    # saving crops pixels df
    crops_df.to_csv(output_path,
                    index=False)

    # printing execution message
    print(f'output saved to {output_path}')
    print('analysis complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting images folder
    images_folder = args_dict['images_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting output path
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running generate_manual_annotations function
    generate_manual_annotations(images_folder=images_folder,
                                images_extension=images_extension,
                                output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
