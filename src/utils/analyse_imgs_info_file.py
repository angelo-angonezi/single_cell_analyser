# analyse imgs info file module

print('initializing...')  # noqa

# Code destined to analysing project's images info file.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from argparse import ArgumentParser
from src.utils.aux_funcs import enter_to_continue
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
    description = "analyse imgs info file module"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input_file param
    input_file_help = 'defines path to input file (imgs_info_file.csv)'
    parser.add_argument('-i', '--input-file',
                        dest='input_file',
                        required=True,
                        help=input_file_help)

    # output_path param
    output_path_help = 'defines path to output csv'
    parser.add_argument('-i', '--output-path',
                        dest='output_path',
                        required=True,
                        help=output_path_help)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def analyse_imgs_info_file(input_file: str,
                           output_path: str
                           ) -> None:
    """
    Given a path to an imgs_info_file csv,
    analyses df based on experimental setups,
    saving summary file in given output path.
    """
    pass

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input file
    input_file = args_dict['input_file']

    # getting output path
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    enter_to_continue()

    # running analyse_imgs_info_file function
    analyse_imgs_info_file(input_file=input_file,
                           output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
