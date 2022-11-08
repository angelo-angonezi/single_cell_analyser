# generate simulated data module

print('initializing...')  # noqa

# code destined to generating random
# simulated data for ml codes testing only.

######################################################################
# importing required libraries

print('importing required libraries...')  # noqa
from time import sleep
from argparse import ArgumentParser
print('all required libraries successfully imported.')  # noqa
sleep(0.8)

#####################################################################
# defining global variables

ITERATIONS_TOTAL = 0
CURRENT_ITERATION = 1
RESIZE_DIMENSIONS = (500, 500)


#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "single cell cropper - tool used to segment cells based on\n"
    description += "machine learning output data.\n"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser
    parser.add_argument('-i', '--images-input-folder',
                        dest='images_input_folder',
                        help='defines path to folder containing images',
                        required=True)

    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        help='defines path to output folder which will contain crops\nand crops info file',
                        required=True)

    parser.add_argument('-d', '--detections-dataframe',
                        dest='detections_df_path',
                        help='defines path to file containing detections info',
                        required=True)

    parser.add_argument('-t', '--detection-threshold',
                        dest='detection_threshold',
                        help='defines detection threshold for ml model results',
                        required=True)

    parser.add_argument('-r', '--resize',
                        dest='resize_toggle',
                        action='store_true',
                        help='resizes all crops to same dimensions',
                        required=False,
                        default=False)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict


######################################################################
# defining auxiliary functions



######################################################################
# defining main function


def main():
    """
    Runs main code.
    """
    # getting data from Argument Parser
    args_dict = get_args_dict()




######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
