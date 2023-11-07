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
from tensorflow.keras.models import load_model
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

# loading the model
print('loading model...')
model = load_model(model_path)

# iterating over images
for file in files:

    # opening current image
    image = imread(file)

    # getting predictions for current image
    predictions_raw = model.predict(image)
    print(file, predictions_raw)

# end of current module
