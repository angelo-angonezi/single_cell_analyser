# predict classes module

print('initializing...')  # noqa

# Code destined to predicting classes
# from single cell crops, using given
# trained neural network.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import DataFrame
from numpy import expand_dims
from argparse import ArgumentParser
from keras.models import load_model
from src.utils.aux_funcs import IMAGE_SIZE
from src.utils.aux_funcs import get_base_df
from src.utils.aux_funcs import load_bgr_img
from src.utils.aux_funcs import is_using_gpu
from src.utils.aux_funcs import resize_image
from src.utils.aux_funcs import enter_to_continue
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = 'predict classes module'

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # images folder param
    parser.add_argument('-i', '--images-folder',
                        dest='images_folder',
                        required=True,
                        help='defines path to folder containing images to be predicted.')

    # images extension param
    parser.add_argument('-e', '--extension',
                        dest='extension',
                        required=True,
                        help='defines images extension (.png, .jpg, .tif).')

    # model path param
    parser.add_argument('-m', '--model-path',
                        dest='model_path',
                        required=True,
                        help='defines path to trained model (.h5 file)')

    # output path param
    parser.add_argument('-o', '--output-path',
                        dest='output_path',
                        required=True,
                        help='defines path to .csv file which will contain predictions.')

    # creating arguments dictionary
    args_dict = vars(parser.parse_args())

    # returning the arguments dictionary
    return args_dict

######################################################################
# defining auxiliary functions


def add_prediction_col(df: DataFrame,
                       model: any,
                       input_folder: str,
                       extension: str
                       ) -> None:
    """
    Given a base df, and a loaded
    model adds new column based on
    model predictions.
    """
    # defining col name
    col_name = 'prediction'

    # getting df rows
    df_rows = df.iterrows()

    # getting rows num
    rows_num = len(df)

    # defining starter for current row index
    current_row_index = 1

    # iterating over df rows
    for row_index, row_data in df_rows:

        # printing execution message
        base_string = f'adding {col_name} col to row #INDEX# #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row image name
        image_name = row_data['crop_name']

        # getting current image name with extension
        image_name_w_extension = f'{image_name}{extension}'

        # getting current image path
        image_path = join(input_folder,
                          image_name_w_extension)

        # opening current image
        current_image = load_bgr_img(image_path=image_path)

        # resizing image
        current_image = resize_image(open_image=current_image,
                                     image_size=IMAGE_SIZE)

        # normalizing current image
        normalized_image = current_image / 255

        # expending dims (required to enter Sequential model)
        expanded_image = expand_dims(normalized_image, 0)

        # getting current image prediction
        current_prediction_list = model.predict(expanded_image,
                                                verbose=0)

        # extracting current prediction from list
        current_prediction = current_prediction_list[0]

        # converting prediction to float
        current_prediction_float = float(current_prediction)

        # updating current row value
        df.at[row_index, col_name] = current_prediction_float

        # updating current row index
        current_row_index += 1


def get_predictions_df(model_path: str,
                       images_folder: str,
                       extension: str,
                       ) -> DataFrame:
    """
    Given a path to a trained model, and a path
    to a folder containing images, runs model
    and returns a predictions data frame of
    following structure:
    |   image    | prediction |
    | img01.jpg  |    3.08    |
    | img02.jpg  |    1.54    |
    ...
    """
    # loading model
    print('loading model...')
    model = load_model(model_path)

    # getting images in folder
    print(f'getting images in folder "{images_folder}"...')
    images_list = get_specific_files_in_folder(path_to_folder=images_folder,
                                               extension=extension)

    # getting images num
    images_num = len(images_list)

    # printing execution message
    f_string = f'{images_num} images found in input folder.'
    print(f_string)

    # removing image extensions
    images_list = [image_name.replace(extension, '')
                   for image_name
                   in images_list]

    # getting base df
    print('getting base df...')
    predictions_df = get_base_df(files=images_list,
                                 col_name='crop_name')

    # adding random predictions col
    print('adding random predictions col...')
    from random import randint
    random_values = [randint(10000, 100000)
                     for _
                     in images_list]
    predictions_df['predictions'] = random_values
    exit()

    # adding predictions col
    print('adding predictions col...')
    add_prediction_col(df=predictions_df,
                       input_folder=images_folder,
                       extension=extension,
                       model=model)

    # returning predictions df
    return predictions_df


def generate_predictions_df(images_folder: str,
                            extension: str,
                            model_path: str,
                            output_path: str
                            ) -> None:
    # getting predictions df
    print('getting predictions df...')
    predictions_df = get_predictions_df(model_path=model_path,
                                        images_folder=images_folder,
                                        extension=extension)

    # saving predictions df
    print('saving predictions df...')
    predictions_df.to_csv(output_path,
                          index=False)

    # printing execution message
    print('analysis complete!')
    print(f'results saved to "{output_path}".')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting images folder param
    images_folder = args_dict['images_folder']

    # getting images extension param
    extension = args_dict['extension']

    # getting model path param
    model_path = args_dict['model_path']

    # getting output path param
    output_path = args_dict['output_path']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # checking gpu usage
    using_gpu = is_using_gpu()
    using_gpu_str = f'Using GPU: {using_gpu}'
    print(using_gpu_str)

    # waiting for user input
    enter_to_continue()

    # running generate_predictions_df function
    generate_predictions_df(images_folder=images_folder,
                            extension=extension,
                            model_path=model_path,
                            output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
