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
from keras.metrics import Recall
from keras.models import load_model
from keras.metrics import Precision
from argparse import ArgumentParser
from keras.metrics import BinaryAccuracy
from src.utils.aux_funcs import is_using_gpu
from src.utils.aux_funcs import normalize_data
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_data_split_from_folder
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

    # splits folder param
    parser.add_argument('-s', '--splits-folder',
                        dest='splits_folder',
                        required=True,
                        help='defines splits folder name (contains "train", "val" and "test" subfolders).')

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
    # defining placeholder values for gts/predictions list
    gts_list = []
    predictions_list = []

    # getting test batches
    test_batches = test_data.as_numpy_iterator()

    # iterating over batches in test data set
    for batch in test_batches:

        # getting current gts and predictions
        current_inputs, current_gts = batch
        current_predictions = model.predict(current_inputs)

        # unpacking values
        gts = [f[0] for f in current_gts]
        predictions = [f[0] for f in current_predictions]

        # appending gts/predictions to respective lists
        for gt in gts:
            gts_list.append(gt)
        for prediction in predictions:
            predictions_list.append(prediction)

    # defining placeholder values for tps, tns, fps, fns
    tps = 0
    tns = 0
    fps = 0
    fns = 0

    # converting values to respective classes (string format)
    gts_list_str = ['excluded' if gt < 0.5 else 'included' for gt in gts_list]
    predictions_list_str = ['excluded' if prediction < 0.5 else 'included' for prediction in predictions_list]

    # zipping lists
    a = zip(gts_list_str, predictions_list_str)
    for i in a:
        gt, prediction = i
        if gt == 'included':
            if prediction == 'included':
                tps += 1
            else:
                fns += 1
        elif gt == 'excluded':
            if prediction == 'excluded':
                tns += 1
            else:
                fps += 1

    # calculating metrics
    accuracy = (tps + tns) / (tps + tns + fps + fns)
    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    # printing results
    f_string = '--Metrics results--\n'
    f_string += f'Accuracy:  {accuracy}\n'
    f_string += f'Precision: {precision}\n'
    f_string += f'Recall:    {recall}\n'
    f_string += f'F1-Score:  {f1_score}'
    print(f_string)


def image_filter_test(splits_folder: str,
                      batch_size: int,
                      model_path: str
                      ) -> None:
    # getting data splits
    print('getting test data...')
    test_data = get_data_split_from_folder(splits_folder=splits_folder,
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

    # getting splits folder param
    splits_folder = args_dict['splits_folder']

    # getting batch size param
    batch_size = int(args_dict['batch_size'])

    # getting model path param
    model_path = args_dict['model_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # checking gpu usage
    using_gpu = is_using_gpu()
    using_gpu_str = f'Using GPU: {using_gpu}'
    print(using_gpu_str)

    # waiting for user input
    enter_to_continue()

    # running image_filter_test function
    image_filter_test(splits_folder=splits_folder,
                      batch_size=batch_size,
                      model_path=model_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
