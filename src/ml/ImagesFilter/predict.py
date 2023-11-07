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
from src.utils.aux_funcs import IMAGE_WIDTH
from src.utils.aux_funcs import IMAGE_HEIGHT
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
print('all required libraries successfully imported.')  # noqa

######################################################################
# defining global variables

model_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\models\\modelV1.h5'
# model_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\models\\modelV1.h5'
data_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\ex'
# data_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\ex'

######################################################################
# running predictions

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
# end of current module
