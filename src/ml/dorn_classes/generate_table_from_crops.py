# generate table from crops module

print('initializing...')  # noqa

# Code destined to generating ML input
# table, based on crops info.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import read_csv
from argparse import ArgumentParser
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
    description = 'generate ML table from crops module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    input_help = 'defines input folder (folder containing images)'
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help=input_help)

    # images_extension param
    images_extension_help = 'defines extension (.tif, .png, .jpg) of images in input folders'
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help=images_extension_help)

    # crops file param
    crops_help = 'defines path to crops file (containing crops info)'
    parser.add_argument('-c', '--crops-file',
                        dest='crops_file',
                        required=True,
                        help=crops_help)

    # output folder param
    output_help = 'defines output folder (folder that will contain outlined images)'
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help=output_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def generate_table_from_crops(input_folder: str,
                              images_extension: str,
                              crops_file: str,
                              output_folder: str,
                              ) -> None:
    """
    Given a path to a folder containing crops,
    and a path to a file containing crops info,
    generates ML input compatible tables
    :param input_folder: String. Represents a path to a folder.
    :param images_extension: String. Represents image extension.
    :param crops_file: String. Represents a path to a file.
    :param output_folder: String. Represents a path to a folder.
    :return: None.
    """
    # getting crops df
    crops_df = read_csv(crops_file)

    # getting images in input folder
    images = get_specific_files_in_folder(path_to_folder=input_folder,
                                          extension=images_extension)
    images_num = len(images)
    images_names = [image_name.replace(images_extension, '')
                    for image_name
                    in images]

    # iterating over images_names
    for image_index, image_name in enumerate(images_names, 1):

        # TODO: check image correspondence between different dfs!

        # printing execution message
        progress_base_string = f'adding overlays to image #INDEX# of #TOTAL#'
        print_progress_message(base_string=progress_base_string,
                               index=image_index,
                               total=images_num)

        # getting image path
        image_name_w_extension = f'{image_name}{images_extension}'
        image_path = join(input_folder, image_name_w_extension)

        # getting output path
        output_name = f'{image_name}_overlays.png'
        output_path = join(output_folder, output_name)

        # adding overlays to current image
        add_overlays_to_single_image(image_name=image_name,
                                     image_path=image_path,
                                     merged_df=merged_df,
                                     detection_threshold=detection_threshold,
                                     output_path=output_path,
                                     color_dict=color_dict,
                                     style=style)

    # printing execution message
    f_string = f'overlays added to all {images_num} images!'
    print(f_string)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input folder
    input_folder = args_dict['input_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting crops file
    crops_file = args_dict['crops_file']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running add_overlays_to_multiple_images function
    generate_table_from_crops(input_folder=input_folder,
                              images_extension=images_extension,
                              crops_file=crops_file,
                              output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
