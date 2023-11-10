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
from os.path import join
from seaborn import lineplot
from pandas import DataFrame
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from src.utils.aux_funcs import IMAGE_WIDTH
from src.utils.aux_funcs import IMAGE_HEIGHT
from src.utils.aux_funcs import is_using_gpu
from src.utils.aux_funcs import normalize_data
from tensorflow.keras.models import Sequential
from src.utils.aux_funcs import get_data_split
from tensorflow.keras.layers import MaxPooling2D
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
    description = 'ImagesFilter train module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # dataset folder param
    parser.add_argument('-d', '--dataset-folder',
                        dest='dataset_path',
                        required=True,
                        help='defines path to folder containing annotated data.')

    # learning rate param
    parser.add_argument('-l', '--learning-rate',
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

    # model path param
    parser.add_argument('-m', '--model-path',
                        dest='model_path',
                        required=True,
                        help='defines path to save model (.h5 file)')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def get_base_model(learning_rate: float):
    # defining model
    print('defining model...')
    model = Sequential()

    # adding layers
    print('adding layers...')
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compiling model
    print('compiling model...')
    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    # printing model summary
    print('printing model summary...')
    model.summary()

    # returning model
    return model


def train_model(model,
                train_data,
                val_data,
                epochs,
                callback
                ):
    print('training model...')
    train_history = model.fit(train_data,
                              epochs=epochs,
                              validation_data=val_data,
                              callbacks=[callback])
    print('training complete!')

    # getting train history dict
    history_dict = train_history.history

    # returning train history
    return history_dict


def save_model(model,
               model_path
               ):
    print('saving model...')
    model.save(model_path)
    print(f'model saved to "{model_path}".')


def get_history_df(history_dict):
    # creating data frame based on dict
    history_df = DataFrame(history_dict)

    # adding epoch column
    history_df['epoch'] = [f for f in range(len(history_df))]

    # melting df
    history_df = history_df.melt('epoch')

    # returning df
    return history_df


def generate_history_plot(dataset_folder: str,
                          df: DataFrame
                          ) -> None:
    # getting save path
    save_path = join(dataset_folder,
                     'plots',
                     'train_history.png')

    # plotting history
    print('plotting history...')
    lineplot(data=df,
             x='epoch',
             y='value',
             hue='variable')

    # saving figure
    fig_path = join(save_path)
    title = 'Train History'
    plt.title(title)
    plt.savefig(fig_path)


def image_filter_train(dataset_path: str,
                       learning_rate: float,
                       epochs: int,
                       batch_size: int,
                       model_path: str
                       ) -> None:
    # getting subfolder paths
    splits_path = join(dataset_path,
                       'splits')

    # getting logdir path
    logdir = join(dataset_path,
                  'logs')

    # getting data splits
    train_data = get_data_split(splits_folder=splits_path,
                                split='train',
                                batch_size=batch_size)
    val_data = get_data_split(splits_folder=splits_path,
                              split='val',
                              batch_size=batch_size)

    # normalizing data to 0-1 scale
    print('normalizing data...')
    train_data = normalize_data(data=train_data)
    val_data = normalize_data(data=val_data)

    # saving example images
    # i = 0
    # for images, labels in test.take(test_size):  # only take first element of dataset
    #     numpy_images = images.numpy()
    #     numpy_labels = labels.numpy()
    #     img = numpy_images[0]
    #     label = numpy_labels[0]
    #     label = label[0]
    #     label = 'excluded' if label == 0.0 else 'included'
    #     img = img.astype(numpy.uint8)
    #     im = Image.fromarray(img)
    #     im.save(join(logdir.replace('logs', 'ex'), f'img{i}_{label}.jpg'))
    #     print(img, label)
    #     i += 1
    # exit()

    # getting model
    model = get_base_model(learning_rate=learning_rate)

    # defining callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # training model (and saving history)
    train_history = train_model(model=model,
                                train_data=train_data,
                                val_data=val_data,
                                epochs=epochs,
                                callback=tensorboard_callback)

    # saving model
    save_model(model=model,
               model_path=model_path)

    # converting history dict to data frame
    history_df = get_history_df(history_dict=train_history)

    # generating history plot
    generate_history_plot(dataset_folder=dataset_path,
                          df=history_df)

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting dataset path param
    dataset_path = str(args_dict['dataset_path'])

    # getting output path param
    learning_rate = float(args_dict['learning_rate'])

    # getting output path param
    epochs = int(args_dict['epochs'])

    # getting batch size param
    batch_size = int(args_dict['batch_size'])

    # getting model path param
    model_path = str(args_dict['model_path'])

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # checking gpu usage
    using_gpu = is_using_gpu()
    using_gpu_str = f'Using GPU: {using_gpu}'
    print(using_gpu_str)

    # waiting for user input
    enter_to_continue()

    # running image_filter_train function
    image_filter_train(dataset_path=dataset_path,
                       learning_rate=learning_rate,
                       epochs=epochs,
                       batch_size=batch_size,
                       model_path=model_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
