# ImagesFilter train module

print('initializing...')  # noqa

# Code destined to training neural network
# to classify images as "included" or "excluded"
# from analyses.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
import tensorflow as tf
from tensorflow import Tensor
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.optimizers import Adam
from argparse import ArgumentParser
from keras.layers import MaxPooling2D
from keras.callbacks import TensorBoard
from keras.applications import ResNet50
from src.utils.aux_funcs import IMAGE_SIZE
from src.utils.aux_funcs import train_model
from src.utils.aux_funcs import is_using_gpu
from keras.engine.sequential import Sequential
from src.utils.aux_funcs import get_history_df
from src.utils.aux_funcs import normalize_data
from keras.applications import InceptionResNetV2
from keras.callbacks import LearningRateScheduler
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import generate_history_plot
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
    description = 'ImagesFilter train module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # splits folder param
    parser.add_argument('-s', '--splits-folder',
                        dest='splits_folder',
                        required=True,
                        help='defines splits folder name (contains "train", "val" and "test" subfolders).')

    # logs folder param
    parser.add_argument('-l', '--logs-folder',
                        dest='logs_folder',
                        required=True,
                        help='defines path to logs folder (will contain train logs and train history plot).')

    # model path param
    parser.add_argument('-m', '--model-path',
                        dest='model_path',
                        required=True,
                        help='defines path to save model (.h5 file)')

    # learning rate param
    parser.add_argument('-lr', '--learning-rate',
                        dest='learning_rate',
                        required=True,
                        help='defines learning rate.')

    # epochs param
    parser.add_argument('-e', '--epochs',
                        dest='epochs',
                        required=True,
                        help='defines number of epochs.')

    # batch size param
    parser.add_argument('-b', '--batch-size',
                        dest='batch_size',
                        required=True,
                        help='defines batch size.')

    # model type param
    parser.add_argument('-t', '--model-type',
                        dest='model_type',
                        required=True,
                        help='defines whether to create new (new) or to use ResNet50 transfer learning model (resnet).')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_resnet_model(input_shape: tuple) -> Sequential:
    """
    Given an input shape, returns
    resnet-based model.
    """
    # defining base model
    model = Sequential()

    # getting resnet base layers
    base_layers = ResNet50(include_top=False,
                           input_shape=input_shape,
                           pooling='max',
                           weights='imagenet')

    # setting resnet layers as untrainable
    for layer in base_layers.layers:
        layer.trainable = False

    # adding resnet layers
    model.add(base_layers)

    # final dense layer
    model.add(Dense(1, activation='sigmoid'))

    # returning model
    return model


def get_inception_model(input_shape: tuple) -> Sequential:
    """
    Given an input shape, returns
    inception-based model.
    """
    # defining base model
    model = Sequential()

    # getting resnet base layers
    base_layers = InceptionResNetV2(include_top=False,
                                    input_shape=input_shape,
                                    pooling='max',
                                    weights='imagenet')

    # setting resnet layers as untrainable
    for layer in base_layers.layers:
        layer.trainable = False

    # adding resnet layers
    model.add(base_layers)

    # final dense layer
    model.add(Dense(1, activation='sigmoid'))

    # returning model
    return model


def get_new_model(input_shape: tuple) -> Sequential:
    """
    Given an input shape, returns
    new self-made model.
    """
    # defining base model
    model = Sequential()

    # defining CNN layers

    # first convolution + pooling (input layer)
    model.add(Conv2D(filters=32,
                     kernel_size=(5, 5),
                     strides=1,
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D())

    # second convolution + pooling
    model.add(Conv2D(filters=16,
                     kernel_size=(3, 3),
                     strides=1,
                     activation='relu'))
    model.add(MaxPooling2D())

    # flattening layer
    model.add(Flatten())

    # mid-dense layers
    model.add(Dense(32, activation='relu'))

    # final dense layer
    model.add(Dense(1, activation='sigmoid'))

    # returning model
    return model


def get_classification_model(learning_rate: float,
                             base_layers: str
                             ) -> Sequential:
    """
    Given a model base and a learning rate,
    returns compiled model.
    """
    # defining model input
    image_height, image_width = IMAGE_SIZE
    input_shape = (image_height, image_width, 3)  # 3 because it is an RGB image

    # getting model
    print('getting model...')

    # defining placeholder value for model
    model = None

    # getting base layers
    if base_layers == 'resnet':

        # getting resnet layers
        model = get_resnet_model(input_shape=input_shape)

    elif base_layers == 'inception':

        # getting inception layers
        models = get_inception_model(input_shape=input_shape)

    else:

        # getting new layers
        model = get_new_model(input_shape=input_shape)

    # defining optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # defining loss function
    loss = 'binary_crossentropy'

    # defining metrics
    metrics = ['accuracy']

    # compiling model
    print('compiling model...')
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    # printing model summary
    print('printing model summary...')
    model.summary()

    # returning model
    return model


def scheduler(epoch: int,
              lr: Tensor
              ) -> Tensor:
    """
    Given an epoch, and a learning
    rate, returns updated learning rate.
    """
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def image_filter_train(splits_folder: str,
                       logs_folder: str,
                       model_path: str,
                       learning_rate: float,
                       epochs: int,
                       batch_size: int,
                       model_type: str
                       ) -> None:
    """
    Trains image filter model.
    """
    # getting data splits
    train_data = get_data_split_from_folder(splits_folder=splits_folder,
                                            split='train',
                                            batch_size=batch_size)
    val_data = get_data_split_from_folder(splits_folder=splits_folder,
                                          split='val',
                                          batch_size=batch_size)

    # printing found classes
    classes_str = f'Classes: {train_data.class_names}'
    print(classes_str)

    # getting model
    model = get_classification_model(learning_rate=learning_rate,
                                     base_layers=model_type)

    # defining callback
    tensorboard_callback = TensorBoard(log_dir=logs_folder)
    lr_callback = LearningRateScheduler(scheduler)

    # training model (and saving history)
    train_history = train_model(model=model,
                                train_data=train_data,
                                val_data=val_data,
                                epochs=epochs,
                                callback=lr_callback)

    # saving model
    print('saving model...')
    model.save(model_path)

    # converting history dict to data frame
    history_df = get_history_df(history_dict=train_history)

    # generating history plot
    generate_history_plot(logs_folder=logs_folder,
                          df=history_df)

    # printing execution message
    print(f'plot/logs saved to "{logs_folder}".')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting splits folder param
    splits_folder = args_dict['splits_folder']

    # getting logs folder param
    logs_folder = args_dict['logs_folder']

    # getting model path param
    model_path = args_dict['model_path']

    # getting output path param
    learning_rate = float(args_dict['learning_rate'])

    # getting output path param
    epochs = int(args_dict['epochs'])

    # getting batch size param
    batch_size = int(args_dict['batch_size'])

    # getting model type param
    model_type = args_dict['model_type']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # checking gpu usage
    using_gpu = is_using_gpu()
    using_gpu_str = f'Using GPU: {using_gpu}'
    print(using_gpu_str)

    # waiting for user input
    enter_to_continue()

    # running image_filter_train function
    image_filter_train(splits_folder=splits_folder,
                       logs_folder=logs_folder,
                       model_path=model_path,
                       learning_rate=learning_rate,
                       epochs=epochs,
                       batch_size=batch_size,
                       model_type=model_type)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
