# ImagesFilter predict module

print('initializing...')  # noqa

# Code destined to classifying images as
# "included" or "excluded" from analyses,
# using previously trained neural network.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os import listdir
from cv2 import imread
from os.path import join
from argparse import ArgumentParser
from src.utils.aux_funcs import IMAGE_WIDTH
from src.utils.aux_funcs import IMAGE_HEIGHT
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
print('all required libraries successfully imported.')  # noqa

######################################################################
# defining global variables

# TODO: add argsparser and main
model_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\models\\modelV1.h5'
# model_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\models\\modelV1.h5'
data_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\ex'
# data_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\ex'
#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'create data set description file module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # annotations file param
    parser.add_argument('-a', '--annotations-file',
                        dest='annotations_file',
                        required=True,
                        help='defines path to fornma nucleus output file (model output format).')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to output csv.')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


# getting data
files = listdir(data_path)

# loading data
print(f'loading data from folder "{data_path}"...')
data = image_dataset_from_directory(directory=data_path,
                                    color_mode='rgb',
                                    batch_size=8,
                                    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                    shuffle=False)

# loading model
model = load_model(model_path)

# getting data batches
data_batches = data.as_numpy_iterator()

# defining placeholder value for predictions
predictions_list = []

# iterating over batches in data set
for batch in data_batches:
    X, y = batch
    current_predictions = model.predict(X)
    current_predictions_list = current_predictions.tolist()
    predictions_list.extend(current_predictions_list)

# removing elements from list in list
predictions_list = [f[0] for f in predictions_list]

print(predictions_list)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting annotations file param
    annotations_file_path = args_dict['annotations_file']

    # getting output path param
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running create_dataset_description_file function
    create_dataset_description_file(annotations_file_path=annotations_file_path,
                                    output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
