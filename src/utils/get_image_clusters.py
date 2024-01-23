# get image clusters module

print('initializing...')  # noqa

# Code destined to obtaining image clusters
# based on feature extracted from the images.

######################################################################
# imports

# importing required libraries
print('importing required libraries...')  # noqa
from umap import UMAP
from os import environ
from os.path import join
from numpy import ndarray
from os.path import exists
from pandas import read_csv
from pandas import DataFrame
from pandas import read_pickle
from keras.models import Model
from keras.utils import load_img
from sklearn.cluster import KMeans
from plotly.express import scatter
from numpy import array as np_array
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from os import makedirs as os_makedirs
from keras.applications.vgg16 import VGG16
from src.utils.aux_funcs import print_gpu_usage
from sklearn.preprocessing import StandardScaler
from src.utils.aux_funcs import enter_to_continue
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.vgg16 import preprocess_input
from src.utils.aux_funcs import print_progress_message
from src.utils.aux_funcs import print_execution_parameters
from src.utils.aux_funcs import get_specific_files_in_folder
from keras.applications.inception_resnet_v2 import InceptionResNetV2
print('all required libraries successfully imported.')  # noqa

# setting tensorflow warnings off
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#####################################################################
# defining global variables

# IMG_WIDTH = 224
IMG_WIDTH = 299
# IMG_HEIGHT = 224
IMG_HEIGHT = 299
SEED = 53
N_COMPONENTS = 10  # defines number of principal components in PCA
N_CLUSTERS = 4  # if set to zero, plots k-means elbow plot and asks user input
N_SAMPLE = 30  # defines number of images per cluster plot
PIXEL_CALC = 'het'  # defines pixel intensity calculation (mean/min/max/het)
# LABEL_COL = 'label'
LABEL_COL = 'class'
# MODEL_NAME = 'vgg16'
MODEL_NAME = 'inception'
# MODEL_NAME = 'resnet'

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

    # labels path param
    labels_help = 'defines path to labels file'
    parser.add_argument('-l', '--labels-path',
                        dest='labels_path',
                        required=True,
                        help=labels_help)

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


def get_base_model(model_name: str) -> Model:
    """
    Returns specified loaded model
    until penultimate layer.
    """
    # printing execution message
    f_string = f'loading "{model_name}" model...'
    print(f_string)

    # defining placeholder value for model
    model = None

    # checking given model name
    if model_name == 'vgg16':

        # loading VGG16
        model = VGG16()

    elif model_name == 'inception':

        # loading InceptionNet
        model = InceptionResNetV2()

    elif model_name == 'resnet':

        # loading ResNet
        model = ResNet50V2()

    else:

        # printing execution message
        f_string = f'model {model_name} not specified.\n'
        f_string += f'Please, check and try again.'
        print(f_string)

        # quitting
        exit()

    # updating model (cutting final layers)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    # returning model
    return model


def get_model_features(file_path: str,
                       model: Model
                       ) -> Model:
    """
    Given a file path, loads image
    and returns extracted features
    based on given model.
    """
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
                             use_multiprocessing=True,
                             verbose=0)

    # returning features
    return features


def get_pixel_intensity(file_path: str,
                        calc: str
                        ) -> float:
    """
    Given a file path, loads image
    and returns pixel intensity value,
    based on given calc method (mean, min, max).
    """
    # loading image
    img = load_img(file_path, target_size=(IMG_WIDTH, IMG_HEIGHT))

    # converting image to numpy array
    img = np_array(img)

    # defining placeholder value for current intensity value
    pixel_intensity = None

    # getting current intensity value based on given calc str

    # calculating min intensity
    if calc == 'min':
        pixel_intensity = img.min()

    # calculating max intensity
    elif calc == 'max':
        pixel_intensity = img.max()

    # calculating mean intensity
    elif calc == 'mean':
        pixel_intensity = img.mean()

    # calculating intensities ratio ('het')
    elif calc == 'het':
        pixel_max = img.max()
        pixel_mean = img.mean()
        pixel_intensity = pixel_max / pixel_mean

    else:

        # printing execution message
        f_string = f'calc mode {calc} not specified.\n'
        f_string += f'Please, check and try again.'
        print(f_string)

        # quitting
        exit()

    # converting pixel intensity to float
    pixel_intensity = float(pixel_intensity)

    # returning pixel intensity value
    return pixel_intensity


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
    # labels_list = [f'C{label}' for label in labels]
    labels_list = [label for label in labels]

    # returning cluster labels
    return labels_list


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


def get_label(file_name: str,
              labels_df: DataFrame,
              label_col: str
              ) -> str:
    """
    Given a file name and a labels df,
    returns label in respective col.
    """
    # filtering df for line matching current file
    file_df = labels_df[labels_df['crop_name'] == file_name]

    # getting current file df row
    file_row = file_df.iloc[0]

    # getting current label
    current_label = file_row[label_col]

    # returning current label
    return current_label


def add_labels_col(df: DataFrame,
                   labels_path: str,
                   images_extension: str
                   ) -> None:
    """
    Given a base image names data frame,
    adds labels col, based on given
    labels path.
    """
    # defining col name
    col_name = 'label'

    # adding placeholder values to col
    df[col_name] = None

    # getting df rows
    df_rows = df.iterrows()

    # getting rows num
    rows_num = len(df)

    # defining starter for current row index
    current_row_index = 1

    # reading labels file
    labels_df = read_csv(labels_path)

    # iterating over rows
    for row in df_rows:

        # printing progress message
        base_string = f'adding label col (row #INDEX# of #TOTAL#)'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row index/data
        row_index, row_data = row

        # getting current row image name
        file_name = row_data['file_name']

        # removing file extension
        file_name = file_name.replace(images_extension, '')

        # getting current row label
        current_label = get_label(file_name=file_name,
                                  labels_df=labels_df,
                                  label_col=LABEL_COL)

        # updating current row col
        df.at[row_index, col_name] = current_label

        # updating current row index
        current_row_index += 1


def add_features_col(df: DataFrame,
                     model_name: str
                     ) -> None:
    """
    Given a base image names data frame,
    adds features column, based on specified
    model features extraction.
    """
    # defining col name
    col_name = f'{model_name}_features'

    # adding placeholder values to col
    df[col_name] = None

    # getting df rows
    df_rows = df.iterrows()

    # getting rows num
    rows_num = len(df)

    # defining starter for current row index
    current_row_index = 1

    # loading base model
    base_model = get_base_model(model_name=model_name)

    # iterating over rows
    for row in df_rows:

        # printing progress message
        base_string = f'adding {model_name} features col (row #INDEX# of #TOTAL#)'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row index/data
        row_index, row_data = row

        # getting current row image path
        file_path = row_data['file_path']

        # getting current feature vector
        current_features = get_model_features(file_path=file_path,
                                              model=base_model)

        # TODO: ADD AREA AND AXIS RATIO HERE

        # updating current row col
        df.at[row_index, col_name] = current_features

        # updating current row index
        current_row_index += 1


def add_pixel_intensity_col(df: DataFrame,
                            calc: str
                            ) -> None:
    """
    Given a base image names data frame,
    adds 'calc' pixel intensity column,
    based on given calc method (mean, min, max).
    """
    # defining col name
    col_name = 'pixel_intensity'

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
        base_string = f'adding {calc} pixel intensity col (row #INDEX# of #TOTAL#)'
        print_progress_message(base_string=base_string,
                               index=current_row_index,
                               total=rows_num)

        # getting current row index/data
        row_index, row_data = row

        # getting current row image path
        file_path = row_data['file_path']

        # getting current pixel intensity
        current_pixel_intensity = get_pixel_intensity(file_path=file_path,
                                                      calc=calc)

        # updating current row col
        df.at[row_index, col_name] = current_pixel_intensity

        # updating current row index
        current_row_index += 1


def create_features_df(input_folder: str,
                       images_extension: str,
                       labels_path: str,
                       model_name: str,
                       calc: str
                       ) -> DataFrame:
    """
    Creates features data frame, based
    on images in given input folder.
    """
    # getting files in input folder
    files = get_specific_files_in_folder(path_to_folder=input_folder,
                                         extension=images_extension)

    # creating df
    features_df = get_base_df(files=files)

    # adding file path col
    add_file_path_col(df=features_df,
                      input_folder=input_folder)

    # adding labels col
    add_labels_col(df=features_df,
                   labels_path=labels_path,
                   images_extension=images_extension)

    # adding mean intensity col
    add_pixel_intensity_col(df=features_df,
                            calc=calc)

    # adding features col
    add_features_col(df=features_df,
                     model_name=model_name)

    # returning features df
    return features_df


def get_features_df(input_folder: str,
                    images_extension: str,
                    labels_path: str,
                    model_name: str,
                    calc: str,
                    output_folder: str
                    ) -> DataFrame:
    """
    Checks whether features data frame
    exists in given output folder, and
    returns loaded df. Creates it from
    base df otherwise.
    """
    # defining placeholder value for features df
    features_df = None

    # defining file name
    file_name = 'features_df.pickle'

    # getting file path
    file_path = join(output_folder,
                     file_name)

    # checking whether file exists
    if exists(file_path):

        # loading features df from existing file
        print('loading features df from existing file...')
        features_df = read_pickle(file_path)

    # if it does not exist
    else:

        # creating features df
        print('creating features df from scratch...')
        features_df = create_features_df(input_folder=input_folder,
                                         images_extension=images_extension,
                                         labels_path=labels_path,
                                         model_name=model_name,
                                         calc=calc)

        # saving features df
        print('saving features df...')
        features_df.to_pickle(file_path)

    # returning features df
    return features_df


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

    # updating cluster values
    clusters_updated = [(cluster + 1) for cluster in clusters]

    # adding values to col
    df[col_name] = clusters_updated


def get_features_array(df: DataFrame,
                       model_name: str
                       ) -> ndarray:
    """
    Given a features data frame, returns
    features array to be used as input for
    further analysis.
    """
    # getting column name
    col_name = f'{model_name}_features'

    # retrieving features col
    features_col = df[col_name].to_numpy()

    # unpacking features arrays
    features_list = [feature_array[0] for feature_array in features_col]

    # converting list to array
    features_array = np_array(features_list)

    # returning features array
    return features_array


def save_cluster_image(df: DataFrame,
                       output_path: str
                       ) -> None:
    """
    Given a cluster data frame,
    saves given images in a single
    figure in given output path.
    """
    # defining figure object
    fig = plt.figure(figsize=(25, 25))

    # getting current cluster sample file paths
    file_paths = df['file_path'].to_list()

    # iterating over images
    for index, file_path in enumerate(file_paths):

        # adding current image in the cluster to plot
        plt.subplot(10, 10, index + 1)

        # loading image
        img = load_img(file_path)

        # converting image to array
        img = np_array(img)

        # adding image to plot
        plt.imshow(img)

        # removing axis
        plt.axis('off')

    # saving current cluster figure
    fig.savefig(output_path)


def generate_image_examples(df: DataFrame,
                            output_folder: str,
                            n_sample: int,
                            group_col: str
                            ) -> None:
    """
    Given a clusters data frame, saves
    clusters/labels example images in
    given output folder (depending on
    given group_col).
    """
    # grouping df by cluster
    df_groups = df.groupby(group_col)

    # getting groups num
    groups_num = len(df_groups)

    # defining starter for current group index
    current_group_index = 1

    # creating subfolder for current group col
    save_folder = join(output_folder,
                       group_col)
    os_makedirs(name=save_folder,
                exist_ok=True)

    # iterating over df groups
    for cluster_id, df_group in df_groups:

        # printing execution message
        base_string = f'generating image for {group_col} #INDEX# of #TOTAL#'
        print_progress_message(base_string=base_string,
                               index=current_group_index,
                               total=groups_num)

        # getting cluster size (current group rows num)
        cluster_size = len(df_group)

        # defining placeholder value for n_sample to be used
        n_sample_used = n_sample

        # checking whether n_sample is larger than cluster size
        if n_sample > cluster_size:

            # updating n_sample_used value to max possible value (equal to cluster size)
            n_sample_used = cluster_size

        # getting current group sample
        df_sample = df_group.sample(n=n_sample_used)

        # defining save name/path
        save_name = f'{group_col}_{cluster_id}.png'
        save_path = join(save_folder,
                         save_name)

        # saving current image
        save_cluster_image(df=df_sample,
                           output_path=save_path)

        # updating current group index
        current_group_index += 1


def get_umap_cols(input_array: ndarray) -> tuple:
    """
    Given a features array, returns
    a tuple with (x, y) values to
    plot UMAP.
    """
    # defining UMAP reducer
    reducer = UMAP()

    # scaling data (converting values to z-scores)
    scaled_data = StandardScaler().fit_transform(input_array)

    # reducing data (fitting UMAP)
    umap_results = reducer.fit_transform(scaled_data)

    # getting x, y values
    x = umap_results[:, 0]
    y = umap_results[:, 1]

    # assembling x, y tuple
    xy_tuple = (x, y)

    # returning x, y tuple
    return xy_tuple


def add_umap_cols(df: DataFrame,
                  model_name: str
                  ) -> None:
    """
    Given a features data frame,
    runs UMAP dimensionality reduction
    and adds respective columns to df.
    """
    # getting features array
    features_array = get_features_array(df=df,
                                        model_name=model_name)

    # getting principal components
    # print(f'getting principal components (running PCA with {N_COMPONENTS} components)...')
    # principal_components = get_principal_components(features=features_array,
    #                                                 n_components=N_COMPONENTS)

    # getting umap coords
    umap_x, umap_y = get_umap_cols(input_array=features_array)

    # adding cols to df
    df['umap_x'] = umap_x
    df['umap_y'] = umap_y


def plot_umap(df: DataFrame,
              output_path: str
              ) -> None:
    """
    Given a features data frame containing
    added umap cols, plots UMAP, coloring
    the plot based on label column.
    !!!INTERACTIVE PLOT!!!
    """
    # converting pixel intensity col to float (allows continuous coloring)
    df['pixel_intensity'] = df['pixel_intensity'].astype(float)

    # plotting UMAP
    fig = scatter(data_frame=df,
                  x='umap_x',
                  y='umap_y',
                  color='pixel_intensity',
                  size='cluster',
                  hover_data='file_name',
                  text='cluster')

    # saving plot
    fig.write_html(output_path)


def get_image_clusters(input_folder: str,
                       images_extension: str,
                       labels_path: str,
                       output_folder: str
                       ) -> None:
    """
    Given a path to a folder containing images,
    gets image clusters based on extracted features,
    saving features and clusters data frames in
    given output folder.
    """
    # getting features df
    print('getting features df...')
    features_df = get_features_df(input_folder=input_folder,
                                  images_extension=images_extension,
                                  labels_path=labels_path,
                                  model_name=MODEL_NAME,
                                  calc=PIXEL_CALC,
                                  output_folder=output_folder)

    # adding UMAP cols
    print('running UMAP (adding UMAP X/Y cols)...')
    add_umap_cols(df=features_df,
                  model_name=MODEL_NAME)

    # getting clusters based on principal components
    print(f'getting clusters based on principal components (running K-Means with {N_CLUSTERS} clusters)...')
    features_array = get_features_array(df=features_df,
                                        model_name=MODEL_NAME)
    clusters_labels = get_clusters_labels(principal_components=features_array,
                                          n_clusters=N_CLUSTERS)

    # adding clusters col
    print('adding clusters col...')
    add_cluster_col(df=features_df,
                    clusters=clusters_labels)

    # plotting UMAP
    print('plotting UMAP...')
    save_name = f'umap.html'
    save_path = join(output_folder,
                     save_name)
    plot_umap(df=features_df,
              output_path=save_path)

    # saving clusters df
    print('saving clusters df...')
    save_name = f'clusters_df.pickle'
    save_path = join(output_folder,
                     save_name)
    features_df.to_pickle(save_path)

    # generating label example images
    print('generating label example images...')
    generate_image_examples(df=features_df,
                            output_folder=output_folder,
                            n_sample=N_SAMPLE,
                            group_col='label')

    # generating cluster example images
    print('generating cluster example images...')
    generate_image_examples(df=features_df,
                            output_folder=output_folder,
                            n_sample=N_SAMPLE,
                            group_col='cluster')

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

    # getting labels path
    labels_path = args_dict['labels_path']

    # getting output folder
    output_folder = args_dict['output_folder']

    # printing execution parameters
    print_execution_parameters(params_dict=args_dict)

    # checking gpu usage
    print_gpu_usage()

    # waiting for user input
    enter_to_continue()

    # running get_image_clusters function
    get_image_clusters(input_folder=input_folder,
                       images_extension=images_extension,
                       labels_path=labels_path,
                       output_folder=output_folder)

######################################################################
# running main function


if __name__ == '__main__':
    main()


######################################################################
# end of current module
