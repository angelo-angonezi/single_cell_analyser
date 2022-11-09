# train dna damage model module

print('initializing...')  # noqa

# code destined to training model
# for dna damage phenotype measurement

######################################################################
# importing required libraries

print('importing required libraries...')  # noqa
import tensorflow
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
                        type=str,
                        help='defines path to folder containing images (.tif)',
                        required=True)

    parser.add_argument('-a', '--annotations-input-file',
                        dest='annotations_input_file',
                        type=str,
                        help='defines path to folder containing annotations (.csv)',
                        required=True)

    parser.add_argument('-o', '--output-folder',
                        dest='output_folder',
                        type=str,
                        help='defines path to output folder',
                        required=True)

    parser.add_argument('-l', '--learning-rate',
                        dest='learning_rate',
                        type=int,
                        help="defines model's learning rate",
                        required=True)

    parser.add_argument('-b', '--batch-size',
                        dest='batch_size',
                        type=int,
                        help="defines model's batch size",
                        required=True)

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def train_dna_damage_model(images_folder_path: str,
                           annotations_file_path: str,
                           output_folder_path: str,
                           learning_rate: int,
                           batch_size: int
                           ) -> None:
    """
    Given parameters, trains model for
    dna damage (nuclear foci count)
    phenotype measurement.
    :param images_folder_path: String. Represents a path to a folder.
    :param annotations_file_path: String. Represents a path to a file.
    :param output_folder_path: String. Represents a path to a folder.
    :param learning_rate: Integer. Represents model's learning rate.
    :param batch_size: Integer. Represents model's batch size.
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

    # getting annotations input folder path
    annotations_file_path = args_dict['annotations_input_file']

    # getting output folder path
    output_folder_path = args_dict['output_folder']

    # getting model's learning rate
    learning_rate = args_dict['learning_rate']

    # getting model's batch size
    batch_size = args_dict['batch_size']

    # printing execution parameters
    p_string = f'---Execution parameters---\n'
    p_string += f'Images input folder: {images_folder_path}\n'
    p_string += f'Annotations file: {annotations_file_path}\n'
    p_string += f'Output folder: {output_folder_path}\n'
    p_string += f'Learning rate: {learning_rate}\n'
    p_string += f'Batch size: {batch_size}\n'
    p_string += f'{"_" * 10}'

    # waiting for user input
    i_string = 'Press "Enter" to continue'
    input(i_string)

    # running train model function
    train_dna_damage_model(images_folder_path=images_folder_path,
                           annotations_file_path=annotations_file_path,
                           output_folder_path=output_folder_path,
                           learning_rate=learning_rate,
                           batch_size=batch_size)

######################################################################
# running main function


if __name__ == '__main__':
    main()

######################################################################
# end of current module
