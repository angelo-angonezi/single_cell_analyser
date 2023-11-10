# ImagesFilter test module

print('initializing...')  # noqa

# Code destined to testing neural network
# to classify images as "included" or "excluded"
# from analyses.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from argparse import ArgumentParser
from tensorflow.keras.metrics import Recall
from src.utils.aux_funcs import is_using_gpu
from src.utils.aux_funcs import get_data_split
from tensorflow.keras.metrics import Precision
from src.utils.aux_funcs import normalize_data
from tensorflow.keras.models import load_model
from src.utils.aux_funcs import enter_to_continue
from tensorflow.keras.metrics import BinaryAccuracy
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
    description = 'ImagesFilter test module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # dataset folder param
    parser.add_argument('-d', '--dataset-folder',
                        dest='dataset_path',
                        required=True,
                        help='defines path to folder containing annotated data.')

    # batch size param
    parser.add_argument('-b', '--batch-size',
                        dest='batch_size',
                        required=True,
                        help='defines batch size.')

    # model path param
    parser.add_argument('-m', '--model-path',
                        dest='model_path',
                        required=True,
                        help='defines path to trained model (.h5 file)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def test_model(model,
               test_data,
               ):
    # starting precision/recall/accuracy instances
    precision = Precision()
    recall = Recall()
    accuracy = BinaryAccuracy()

    # getting test batches
    test_batches = test_data.as_numpy_iterator()

    # iterating over batches in test data set
    for batch in test_batches:
        current_input, y = batch
        yhat = model.predict(current_input)
        precision.update_state(y, yhat)
        recall.update_state(y, yhat)
        accuracy.update_state(y, yhat)

    # getting results
    precision_result = precision.result()
    recall_result = recall.result()
    accuracy_result = accuracy.result()

    # printing results
    print('Precision: ', precision_result)
    print('Recall: ', recall_result)
    print('Accuracy: ', accuracy_result)


def image_filter_test(dataset_path: str,
                      batch_size: int,
                      model_path: str
                      ) -> None:
    # getting subfolder paths
    print('getting data path...')
    splits_path = join(dataset_path,
                       'splits')

    # getting data splits
    print('getting test data...')
    test_data = get_data_split(splits_folder=splits_path,
                               split='test',
                               batch_size=batch_size)

    # normalizing data to 0-1 scale
    print('normalizing data...')
    test_data = normalize_data(data=test_data)

    # loading model
    print('loading model...')
    model = load_model(model_path)

    # testing model on test split
    print('testing model...')
    test_model(model=model,
               test_data=test_data)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting output path param
    dataset_path = str(args_dict['dataset_path'])

    # getting batch size param
    batch_size = str(args_dict['batch_size'])

    # getting output path param
    model_path = str(args_dict['model_path'])

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # checking gpu usage
    using_gpu = is_using_gpu()
    using_gpu_str = f'Using GPU: {using_gpu}'
    print(using_gpu_str)

    # waiting for user input
    enter_to_continue()

    # running image_filter_test function
    image_filter_test(dataset_path=dataset_path,
                      batch_size=batch_size,
                      model_path=model_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
