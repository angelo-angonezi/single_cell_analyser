# analyse compare annotations module

print('initializing...')  # noqa

# Code destined to analysing "compare_annotations.py" module
# output (a csv file containing precision and recall data)

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from pandas import concat
from pandas import read_csv
from pandas import DataFrame
from argparse import ArgumentParser
from src.utils.aux_funcs import spacer
from src.utils.aux_funcs import flush_or_print
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "analyse compare_annotations.py output module"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input file param
    input_help = 'defines path to compare_annotations.py output[.csv]'
    parser.add_argument('-i', '--input-file',
                        dest='input_file',
                        required=True,
                        help=input_help)

    # output file param
    output_help = 'defines output folder (folder that will contain output csvs)'
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



######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input file
    input_file = args_dict['input_file']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    f_string = f'--Execution parameters--\n'
    f_string += f'input file: {input_file}\n'
    f_string += f'output folder: {output_folder}'
    spacer()
    print(f_string)
    spacer()
    input('press "Enter" to continue')

    # running

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
