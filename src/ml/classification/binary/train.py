# binary classification train module

print('initializing...')  # noqa

# Code destined to training binary
# classification neural network.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os import environ
from os.path import join
from pandas import read_csv
from tensorflow import Tensor
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.regularizers import l1
from keras.regularizers import l2
from keras.layers import MaxPool2D
from keras.regularizers import l1_l2
from argparse import ArgumentParser
from keras.optimizers import RMSprop
from keras.applications import VGG16
from keras.layers import MaxPooling2D
from keras.callbacks import TensorBoard
from keras.applications import ResNet50
from keras.metrics import BinaryAccuracy
from tensorflow.math import exp as tf_exp
from keras.applications import ConvNeXtTiny
from keras.layers import BatchNormalization
from keras.losses import BinaryCrossentropy
from src.utils.aux_funcs import INPUT_SHAPE
from src.utils.aux_funcs import train_model
from src.utils.aux_funcs import is_using_gpu
from keras.engine.sequential import Sequential
from src.utils.aux_funcs import get_history_df
from keras.layers import GlobalAveragePooling2D
from keras.applications import InceptionResNetV2
from keras.callbacks import LearningRateScheduler
from src.utils.aux_funcs import enter_to_continue
from tensorflow.keras.callbacks import EarlyStopping
from src.utils.aux_funcs import generate_history_plot
from src.utils.aux_funcs import get_data_split_from_df
from src.utils.aux_funcs import print_execution_parameters
print('all required libraries successfully imported.')  # noqa

# setting tensorflow warnings off
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'binary classification train module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # dataset file param
    parser.add_argument('-d', '--dataset-file',
                        dest='dataset_file',
                        required=True,
                        help='defines path to dataset df (.csv) file')

    # splits folder param
    parser.add_argument('-s', '--splits-folder',
                        dest='splits_folder',
                        required=True,
                        help='defines splits folder name (contains "train", "val" and "test" subfolders).')

    # images extension param
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help='defines extension (.tif, .png, .jpg) of images in input folder')

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

    # getting base model
    base_model = ResNet50(include_top=False,
                          input_shape=input_shape,
                          pooling='max',
                          weights='imagenet')

    # getting base layers
    base_layers = base_model.layers
    base_layers_num = len(base_layers)

    # printing execution message
    f_string = f'Num of layers in ResNet based architecture: {base_layers_num}'
    print(f_string)

    # iterating over layers
    for layer_index, layer in enumerate(base_layers):

        # checking layer index
        if layer_index < 20:

            # freezing layer
            layer.trainable = False

        # adding layer to model
        model.add(layer)

    # adding dense/dropout layers
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(rate=0.5))

    # final dense layer
    model.add(Dense(units=1, activation='sigmoid'))

    # returning model
    return model


def get_vgg_model(input_shape: tuple) -> Sequential:
    """
    Given an input shape, returns
    vgg16-based model.
    """
    # defining base model
    model = Sequential()

    # getting base model
    base_model = VGG16(include_top=False,
                       input_shape=input_shape,
                       pooling='max',
                       weights='imagenet')

    # getting base layers
    base_layers = base_model.layers
    base_layers_num = len(base_layers)

    # printing execution message
    f_string = f'Num of layers in VGG based architecture: {base_layers_num}'
    print(f_string)

    # iterating over layers
    for layer_index, layer in enumerate(base_layers):

        # checking layer index
        if layer_index < 10:

            # freezing layer
            layer.trainable = False

        # adding layer to model
        model.add(layer)

    # defining regularizers
    regularizer_l1 = l1(0.001)
    regularizer_l2 = l2(0.01)
    regularizer_l1_l2 = l1_l2(l1=0.01,
                              l2=0.001)

    # mid-dense + dropout layers
    # model.add(Dense(units=512,
    #                 activation='relu',
    #                 kernel_regularizer=regularizer_l2))
    # model.add(Dropout(rate=0.5))
    model.add(Dense(units=256,
                    activation='relu',
                    kernel_regularizer=regularizer_l2))
    model.add(Dropout(rate=0.5))

    # final dense layer
    model.add(Dense(units=1, activation='sigmoid'))

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
    model.add(Conv2D(filters=16,
                     kernel_size=(3, 3),
                     strides=1,
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D())

    # first convolution + pooling (input layer)
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     strides=1,
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D())

    # second convolution + pooling
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     strides=1,
                     activation='relu'))
    model.add(MaxPooling2D())

    # flattening layer
    model.add(Flatten())

    # mid-dense + dropout layers
    model.add(Dense(units=3200, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1600, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=800, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=400, activation='relu'))

    # final dense layer
    model.add(Dense(units=1, activation='sigmoid'))

    # returning model
    return model


def get_convnext_model(input_shape: tuple) -> Sequential:
    """
    Given an input shape, returns
    convnext-based model.
    """
    # defining base model
    model = Sequential()

    # getting base model
    base_model = ConvNeXtTiny(include_top=False,
                              input_shape=input_shape,
                              pooling='max',
                              weights='imagenet')

    # getting base layers
    base_layers = base_model.layers
    base_layers_num = len(base_layers)

    # printing execution message
    f_string = f'Num of layers in ConvNext based architecture: {base_layers_num}'
    print(f_string)

    # adding layers to model
    model.add(base_model)

    # mid-dense + dropout layers
    model.add(Dense(units=384,
                    activation='relu'))
    model.add(Dropout(rate=0.5))

    # final dense layer
    model.add(Dense(units=1, activation='sigmoid'))

    # returning model
    return model


def get_classification_model(input_shape: tuple,
                             learning_rate: float,
                             model_type: str
                             ) -> Sequential:
    """
    Given a model base and a learning rate,
    returns compiled model.
    """
    # defining placeholder value for model
    model = None

    # getting base layers
    print('getting base layers...')

    if model_type == 'resnet':

        # getting resnet layers
        model = get_resnet_model(input_shape=input_shape)

    elif model_type == 'convnext':

        # getting convnext layers
        model = get_convnext_model(input_shape=input_shape)

    elif model_type == 'vgg':

        # getting vgg layers
        model = get_vgg_model(input_shape=input_shape)

    else:

        # getting new layers
        model = get_new_model(input_shape=input_shape)

    # defining optimizer
    optimizer = Adam(learning_rate=learning_rate)

    # defining loss function
    loss = BinaryCrossentropy()

    # defining metrics
    metrics = [BinaryAccuracy()]

    # compiling model
    print('compiling model...')
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    # printing model summary
    print('printing model summary...')
    model.summary()

    # waiting for user input
    enter_to_continue()

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
        return lr * tf_exp(-0.1)


def binary_classification_train(splits_folder: str,
                                extension: str,
                                dataset_file: str,
                                logs_folder: str,
                                model_path: str,
                                learning_rate: float,
                                epochs: int,
                                batch_size: int,
                                model_type: str,
                                input_shape: tuple
                                ) -> None:
    """
    Trains binary classification model.
    """
    # reading dataset df
    print('reading dataset df...')
    dataset_df = read_csv(dataset_file)

    # getting data splits
    print('getting data splits...')
    train_data = get_data_split_from_df(splits_folder=splits_folder,
                                        extension=extension,
                                        dataset_df=dataset_df,
                                        split='train',
                                        batch_size=batch_size,
                                        class_mode='binary')
    val_data = get_data_split_from_df(splits_folder=splits_folder,
                                      extension=extension,
                                      dataset_df=dataset_df,
                                      split='val',
                                      batch_size=batch_size,
                                      class_mode='binary')

    # waiting for user input
    enter_to_continue()

    # getting model
    print('getting model...')
    model = get_classification_model(input_shape=input_shape,
                                     learning_rate=learning_rate,
                                     model_type=model_type)

    # defining callback
    tensorboard_callback = TensorBoard(log_dir=logs_folder)
    # lr_callback = LearningRateScheduler(scheduler)
    # early_stopping = EarlyStopping(monitor='val_binary_accuracy',
    #                                patience=5)

    # training model (and saving history)
    train_history = train_model(model=model,
                                train_data=train_data,
                                val_data=val_data,
                                epochs=epochs,
                                callback=tensorboard_callback)

    # saving model
    print('saving model...')
    model.save(model_path)

    # converting history dict to data frame
    history_df = get_history_df(history_dict=train_history)

    # saving history df
    print('saving train history df...')
    save_name = 'train_history.csv'
    save_path = join(logs_folder,
                     save_name)
    history_df.to_csv(save_path,
                      index=False)

    # generating history plot
    generate_history_plot(logs_folder=logs_folder,
                          df=history_df,
                          epochs_num=epochs)

    # printing execution message
    print(f'model saved to "{model_path}".')
    print(f'plot/logs saved to "{logs_folder}".')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting splits folder param
    splits_folder = args_dict['splits_folder']

    # getting images extension param
    extension = args_dict['images_extension']

    # getting dataset file param
    dataset_file = args_dict['dataset_file']

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
    binary_classification_train(splits_folder=splits_folder,
                                extension=extension,
                                dataset_file=dataset_file,
                                logs_folder=logs_folder,
                                model_path=model_path,
                                learning_rate=learning_rate,
                                epochs=epochs,
                                batch_size=batch_size,
                                model_type=model_type,
                                input_shape=INPUT_SHAPE)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
