# image format converter (* -> 8bit)
######################################################################
# imports

# importing required libraries
from cv2 import imread
from cv2 import imwrite
from os.path import join
from numpy import ndarray
from numpy import uint8 as np_uint8
from argparse import ArgumentParser
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import get_specific_files_in_folder

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "convert images to 8bit module"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    i_help = 'defines path to folder containing input (16-bit .tif) images'
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help=i_help)

    # output folder param
    o_help = 'defines path to folder which will contain output (8-bit) images'
    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        required=True,
                        help=o_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def convert_img_scale(img: ndarray,
                      target_type_min: int,
                      target_type_max: int,
                      target_type: type):
    # getting current image min/max
    img_min = img.min()
    img_max = img.max()

    # getting conversion factors
    a = (target_type_max - target_type_min) / (img_max - img_min)
    b = target_type_max - a * img_max

    # converting image
    converted_img = (a * img + b).astype(target_type)

    # returning converted image
    return converted_img


def convert_single_file(input_file_path: str,
                        output_file_path: str
                        ) -> None:
    """
    Given a path to an image, opens image and
    saves in given output path, converted to 8-bit.
    """
    # opening image
    img = imread(input_file_path, -1)

    # converting image to 8bit
    img_8bit = convert_img_scale(img=img,
                                 target_type_min=0,
                                 target_type_max=256,
                                 target_type=np_uint8)

    # saving image in output folder as 8bit
    imwrite(output_file_path,
            img_8bit)


def convert_multiple_files(input_folder_path: str,
                           output_folder_path: str
                           ) -> None:
    """
    Given a path to a folder containing images,
    converts images to 8bit format, saving them
    in output folder.
    """
    # getting files in input folder
    files = get_specific_files_in_folder(path_to_folder=input_folder_path,
                                         extension='.tif')  # since only tif supports 16-bit
    files_num = len(files)

    # iterating over files in input folder
    for file_index, file in enumerate(files, 1):

        # printing execution message
        f_string = f'converting image #INDEX# of #TOTAL#'
        print_progress_message(base_string=f_string,
                               index=file_index,
                               total=files_num)

        # getting input file path
        input_file_path = join(input_folder_path, file)

        # getting output file path
        output_file_path = join(output_folder_path, file)

        # converting image
        convert_single_file(input_file_path=input_file_path,
                            output_file_path=output_file_path)

    # printing execution message
    f_string = f'all {files_num} files converted!'
    print(f_string)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input file
    input_folder = args_dict['input_folder']

    # getting output path
    output_folder = args_dict['output_folder']

    # running multiple converter function
    convert_multiple_files(input_folder_path=input_folder,
                           output_folder_path=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
