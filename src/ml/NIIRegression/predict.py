# NIIRegressor predict module

print('initializing...')  # noqa

# Code destined to predicting NII values
# from single cell crops, using NIIRegressor
# neural network.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from pandas import concat
from pandas import DataFrame
from numpy import expand_dims
from argparse import ArgumentParser
from keras.models import load_model
from numpy import float16 as np_float  # good to prevent memory crashes
from src.utils.aux_funcs import IMAGE_SIZE
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
    description = 'NIIRegressor predict module'

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

    # defining placeholder value for dfs list
    dfs_list = []

    # getting images num
    images_num = len(images_list)

    # defining dict types
    dict_types = {'image_name': str,
                  'prediction': np_float}

    # defining starter for current index
    current_index = 1

    # iterating over images in images list
    for image in images_list:

        # printing progress message
        base_string = 'analysing image #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_index,
                               total=images_num)

        # getting current image name
        image_name = image.replace(extension, '')

        # getting current image path
        current_path = join(images_folder,
                            image)

        # opening current image
        current_image = load_bgr_img(image_path=current_path)

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

        # assembling current image dict
        current_dict = {'image_name': image_name,
                        'prediction': current_prediction_float}

        # assembling current image df
        current_df = DataFrame(current_dict,
                               index=[0])
        print(current_df)

        # converting df types
        current_df = current_df.astype(dict_types)
        print(current_df)
        exit()

        # appending current df to dfs list
        dfs_list.append(current_df)

        # updating current index
        current_index += 1

    # concatenating dfs in dfs list
    final_df = concat(dfs_list,
                      ignore_index=True)

    # returning final df
    return final_df


def nii_regression_predict(images_folder: str,
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

    # running nii_regression_predict function
    nii_regression_predict(images_folder=images_folder,
                           extension=extension,
                           model_path=model_path,
                           output_path=output_path)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
