# analyse imgs info file module

print('initializing...')  # noqa

# Code destined to analysing project's images info file.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from numpy import intp
from cv2 import imread
from cv2 import imwrite
from cv2 import putText
from cv2 import cvtColor
from os.path import join
from numpy import ndarray
from pandas import concat
from pandas import Series
from cv2 import boxPoints
from pandas import read_csv
from pandas import DataFrame
from cv2 import drawContours
from cv2 import COLOR_BGR2RGB
from cv2 import COLOR_RGB2BGR
from argparse import ArgumentParser
from cv2 import FONT_HERSHEY_SIMPLEX
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import flush_or_print
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "analyse imgs info file module"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input file param
    input_help = 'defines path to input file (imgs_info_file.csv)'
    parser.add_argument('-i', '--input-file',
                        dest='input_file',
                        required=True,
                        help=input_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions




######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input file
    input_file = args_dict['input_file']

    # printing execution parameters
    f_string = f'--Execution parameters--\n'
    f_string += f'input file: {input_file}'
    spacer()
    print(f_string)
    spacer()
    input('press "Enter" to continue')
    spacer()

    # running generate_ function
    add_overlays_to_multiple_images(input_folder=input_folder,
                                    images_extension=images_extension,
                                    detection_file_path=detection_file,
                                    ground_truth_file_path=ground_truth_file,
                                    output_folder=output_folder,
                                    detection_threshold=detection_threshold,
                                    color_dict=COLOR_DICT)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
