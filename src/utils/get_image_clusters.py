# get image clusters module

print('initializing...')  # noqa

# Code destined to obtaining image clusters
# based on feature extracted from the images.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from os.path import join
from random import sample
from numpy import ndarray
from pandas import DataFrame
from keras.models import Model
from keras.utils import load_img
from sklearn.cluster import KMeans
from numpy import array as np_array
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from keras.utils import img_to_array
from sklearn.decomposition import PCA
from keras.applications.vgg16 import VGG16
from src.utils.aux_funcs import is_using_gpu
from src.utils.aux_funcs import get_axis_ratio
from src.utils.aux_funcs import enter_to_continue
from keras.applications.vgg16 import preprocess_input
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
print('all required libraries successfully imported.')  # noqa

#####################################################################
# defining global variables

IMG_WIDTH = 224
IMG_HEIGHT = 224
SEED = 53
N_COMPONENTS = 2
N_CLUSTERS = 10

#####################################################################
# argument parsing related functions


def get_args_dict() -> dict:
    """
    Parses the arguments and returns a dictionary of the arguments.
    :return: Dictionary. Represents the parsed arguments.
    """
    # defining program description
    description = "gets image clusters based on features extracted from images"

    # creating a parser instance
    parser = ArgumentParser(description=description)

    # adding arguments to parser

    # input folder param
    input_help = 'defines input folder (folder containing images)'
    parser.add_argument('-i', '--input-folder',
                        dest='input_folder',
                        required=True,
                        help=input_help)

    # image extension param
    extension_help = 'defines extension (.tif, .png, .jpg) of images in input folder'
    parser.add_argument('-x', '--images-extension',
                        dest='images_extension',
                        required=True,
                        help=extension_help)

    # output folder param
    output_help = 'defines output folder'
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


def get_base_model() -> Model:
    """
    Returns loaded VGG16 model
    until penultimate layer.
    """
    # defining model
    model = VGG16()

    # updating model (cutting final layers)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    # returning model
    return model


def get_vgg_features(file_path: str,
                     model: Model
                     ) -> Model:
    # loading image
    img = load_img(file_path, target_size=(IMG_WIDTH, IMG_HEIGHT))

    # converting image to numpy array
    img = np_array(img)

    # reshaping data for the model reshape(num_of_samples, width, height, channels)
    reshaped_img = img.reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)

    # prepare image for model
    processed_image = preprocess_input(reshaped_img)

    # get the feature vector
    features = model.predict(processed_image,
                             use_multiprocessing=True)

    # returning features
    return features


def get_features_dict(files: list,
                      folder: str
                      ) -> dict:
    """
    Given a list of file paths, returns a
    dictionary in which keys are file names,
    and values are feature vectors.
    """
    # defining placeholder value for features dict
    features_dict = {}

    # getting files total
    files_total = len(files)

    # defining starter for current image index
    current_file = 1

    # defining base model
    base_model = get_base_model()

    # iterating over files
    for file in files:

        # printing execution message
        base_string = f'getting feature vector for image #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_file,
                               total=files_total)

        # getting current file path
        file_path = join(folder, file)

        # getting current image feature vector
        current_feature_vector = get_vgg_features(file_path=file_path,
                                                  model=base_model)

        # assembling current dict element
        current_dict = {file: current_feature_vector}

        # updating features dict
        features_dict.update(current_dict)

        # updating current image index
        current_file += 1

    # returning features dict
    return features_dict


def get_principal_components(features: ndarray,
                             n_components: int
                             ) -> ndarray:
    """
    Given an array of feature vectors,
    returns N number of principal
    components, according to PCA.
    """
    # getting PCA based on given components num
    pca = PCA(n_components=n_components,
              random_state=SEED)

    # fitting PCA
    pca.fit(features)

    # getting principal components
    principal_components = pca.transform(features)

    # returning principal components
    return principal_components


def get_clusters_labels(principal_components: ndarray,
                        n_clusters: int
                        ) -> list:
    # defining base clustering algorithm
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=SEED,
                    n_init=10)

    # fitting clusters
    kmeans.fit(principal_components)

    # getting clusters labels
    labels = kmeans.labels_

    # converting cluster labels to list
    labels_list = [f for f in labels]

    # returning cluster labels
    return labels_list


def get_clusters_dict(files: list,) -> dict:
    # defining placeholder value for clusters dict
    clusters_dict = {}
    pass

    # getting files/labels zip
    clusters_zip = zip(files, labels)

    # iterating over clusters zip
    for file, cluster in clusters_zip:

        # checking if current cluster key already exists
        if cluster not in clusters_dict.keys():

            # adding current cluster id as key
            clusters_dict[cluster] = []

        # updating clusters dict (adding current file to existing files list)
        clusters_dict[cluster].append(file)

    # returning clusters dict
    return clusters_dict


def get_base_df(files: list) -> DataFrame:
    """
    Given a list of files, returns base
    data frame, used on following analysis.
    """
    # defining col name
    col_name = 'file_name'

    # assembling new col
    new_col = {col_name: files}

    # creating data frame
    base_df = DataFrame(new_col)

    # returning base df
    return base_df


def add_file_path_col(df: DataFrame,
                      input_folder: str
                      ) -> None:
    """
    Given a base image names data frame,
    adds file path column, based on given
    input folder.
    """
    # defining col name
    col_name = 'file_path'

    # adding placeholder values to col
    df[col_name] = None

    # getting df rows
    df_rows = df.iterrows()

    # getting rows num
    rows_num = len(df)

    # defining starter for current row index
    current_row_index = 1

    # iterating over rows
    for row in df_rows:

        # printing progress message
        base_string = f'adding file path col (row #INDEX# of #TOTAL#)'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row index/data
        row_index, row_data = row

        # getting current row image name
        file_name = row_data['file_name']

        # getting current row file path
        current_file_path = join(input_folder,
                                 file_name)

        # updating current row col
        df.at[row_index, col_name] = current_file_path

        # updating current row index
        current_row_index += 1


def add_features_col(df: DataFrame) -> None:
    """
    Given a base image names data frame,
    adds features column, based on VGG16
    features extraction.
    """
    # defining col name
    col_name = 'vgg_features'

    # adding placeholder values to col
    df[col_name] = None

    # getting df rows
    df_rows = df.iterrows()

    # getting rows num
    rows_num = len(df)

    # defining starter for current row index
    current_row_index = 1

    # loading base model
    base_model = get_base_model()

    # iterating over rows
    for row in df_rows:

        # printing progress message
        base_string = f'adding vgg features col (row #INDEX# of #TOTAL#)'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row index/data
        row_index, row_data = row

        # getting current row image path
        file_path = row_data['file_path']

        # getting current feature vector
        current_features = get_vgg_features(file_path=file_path,
                                            model=base_model)

        # updating current row col
        df.at[row_index, col_name] = current_features

        # updating current row index
        current_row_index += 1


def add_cluster_col(df: DataFrame,
                    clusters: list
                    ) -> None:
    """
    Given a base image names data frame,
    adds clusters column, based on given
    clusters list.
    """
    # defining col name
    col_name = 'cluster'

    # adding values to col
    df[col_name] = clusters


def generate_cluster_examples(df: DataFrame,
                              output_folder: str,
                              n_sample: int
                              ) -> None:
    """
    Given a clusters data frame, saves
    clusters example images in given
    output folder.
    """
    # grouping df by cluster
    df_groups = df.groupby('cluster')

    # getting groups num
    groups_num = len(df_groups)

    # defining starter for current group index
    current_group_index = 1

    # iterating over df groups
    for cluster_id, df_group in df_groups:

        # printing execution message
        base_string = 'generating image for cluster #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_group_index,
                               total=groups_num)

        # getting current group sample
        # TODO: finish this function

        # saving current image

        # updating current group index
        current_group_index += 1


def get_image_clusters(input_folder: str,
                       images_extension: str,
                       output_folder: str
                       ) -> None:
    """
    Given a path to a folder containing images,
    gets image clusters based on extracted features,
    saving features and clusters data frames in
    given output folder.
    """
    # getting images in input folder
    print('getting images in input folder...')
    files = get_specific_files_in_folder(path_to_folder=input_folder,
                                         extension=images_extension)

    # getting images num
    images_num = len(files)

    # printing execution message
    f_string = f'found {images_num} images in input folder.'
    print(f_string)

    # getting initial data frame based on image names
    print('getting base df...')
    df = get_base_df(files=files)

    # adding file path col
    print('adding file path col...')
    add_file_path_col(df=df,
                      input_folder=input_folder)

    # adding features col
    print('adding features col...')
    add_features_col(df=df)

    # retrieving features col
    features_col = df['vgg_features'].to_numpy()

    # unpacking features arrays
    features_list = [feature_array[0] for feature_array in features_col]

    # converting list to array
    features_arr = np_array(features_list)

    # reshaping features  so that there are N samples of 4096 vectors
    features = features_arr.reshape(-1, 4096)

    # getting principal components
    print(f'getting principal components (running PCA with {N_COMPONENTS} components)...')
    principal_components = get_principal_components(features=features,
                                                    n_components=N_COMPONENTS)

    # getting clusters based on principal components
    print(f'getting clusters based on principal components (running K-Means with {N_CLUSTERS} clusters)...')
    clusters_labels = get_clusters_labels(principal_components=principal_components,
                                          n_clusters=N_CLUSTERS)

    # adding clusters col
    print('adding clusters col...')
    add_cluster_col(df=df,
                    clusters=clusters_labels)

    # saving clusters df
    print('saving clusters df...')
    save_name = f'clusters_df.pickle'
    save_path = join(output_folder,
                     save_name)
    df.to_pickle(save_path)

    # generating cluster example images
    print('generating cluster example images...')
    generate_cluster_examples(df=df,
                              output_folder=output_folder)

    # printing execution message
    print('clustering complete!')

######################################################################
# defining main function


def main():
    """Runs main code."""
    # getting args dict
    args_dict = get_args_dict()

    # getting input folder
    input_folder = args_dict['input_folder']

    # getting image extension
    images_extension = args_dict['images_extension']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # waiting for user input
    # enter_to_continue()

    # running get_image_clusters function
    get_image_clusters(input_folder=input_folder,
                       images_extension=images_extension,
                       output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
