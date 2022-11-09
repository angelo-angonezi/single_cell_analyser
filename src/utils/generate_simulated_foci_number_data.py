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
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "generate simulated data for single-cell crops ML\n"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser
    parser.add_argument('-i', '--images-input-folder',
                        dest='images_input_folder',
                        help='defines path to folder containing images',
                        required=True)

    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        help='defines path to output csv file containing simulated data',
                        required=True)

    parser.add_argument('-n', '--foci-min',
                        dest='foci_num_min',
                        help='defines minimum number of foci in simulated data',
                        required=True)

    parser.add_argument('-m', '--foci-max',
                        dest='foci_num_max',
                        help='defines maximum number of foci in simulated data',
                        required=True)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict


######################################################################
# defining auxiliary functions


def simulate_foci_data(images_folder_path: str,
                       output_path: str,
                       foci_num_min: int,
                       foci_num_max: int
                       ) -> None:
    """
    Given a path to a folder containing images,
    a path to an output file which will contain
    simulated data (csv format), and foci minimum
    and maximum numbers, generates simulated
    data for ml purposes.
    :param images_folder_path: String. Represents a path to a folder.
    :param output_path: String. Represents a path to a file.
    :param foci_num_min: Integer. Represents number of foci in a given image.
    :param foci_num_max: Integer. Represents number of foci in a given image.
    :return: None.
    """
    pass

######################################################################
# defining main function


def main():
    """
    Runs main code.
    """
    # getting data from Argument Parser
    args_dict = get_args_dict()

    # getting images input folder path
    images_folder_path = args_dict['images_input_folder']

    # getting output path
    output_path = args_dict['output_path']

    # getting foci number min
    foci_num_min = args_dict['foci_num_min']

    # getting foci number max
    foci_num_max = args_dict['foci_num_max']

    # running data simulation function
    simulate_foci_data(images_folder_path=images_folder_path,
                       output_path=output_path,
                       foci_num_min=foci_num_min,
                       foci_num_max=foci_num_max)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
