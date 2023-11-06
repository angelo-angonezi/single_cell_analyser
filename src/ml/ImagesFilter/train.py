# ImagesFilter train module

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
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import image_dataset_from_directory
print('all required libraries successfully imported.')  # noqa

######################################################################
# defining global variables
data_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\imgs\\dataset'
# data_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\imgs\\dataset'
logdir = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\logs'
# logdir = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\logs'
save_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\models\\modelV1.h5'
# save_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\models\\modelV1.h5'
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
epochs = 3

######################################################################
# running training

# loading data
print(f'loading data from folder "{data_path}"...')
data = image_dataset_from_directory(data_path)

# normalizing data to 0-1 scale
print('normalizing data...')
data = data.map(lambda x, y: (x/255, y))

# creating data iterator
print('creating data iterator...')
data_iterator = data.as_numpy_iterator()

# getting batches
print('getting data batch...')
batch = data_iterator.next()

# getting split sizes
train_size = int(len(data) * train_ratio)
val_size = int(len(data) * val_ratio)
test_size = int(len(data) * test_ratio)

# getting data splits
print('getting data splits...')
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
f_string = f'Train: {train_ratio * 100}%\n'
f_string += f'Val: {val_ratio * 100}%\n'
f_string += f'Test: {test_ratio * 100}%'
print(f_string)

# defining model
print('defining model...')
model = Sequential()

# adding layers
print('adding layers...')
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
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
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

# printing model summary
print('printing model summary...')
model.summary()

# defining callback
print('defining callback...')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# training model (and saving history)
print('training model...')
train_history = model.fit(train,
                          epochs=epochs,
                          validation_data=val,
                          callbacks=[tensorboard_callback])

# saving model
print('saving model...')
model.save(save_path)
print(f'model saved to "{save_path}".')

# converting history dict to data frame
print('getting training history dict...')
history_dict = train_history.history
history_df = DataFrame(history_dict)
history_df['epoch'] = [f for f in range(len(history_df))]
history_df.head()
history_df = history_df.melt('epoch')

# plotting history
print('plotting history...')
lineplot(data=history_df,
         x='epoch',
         y='value',
         hue='variable')

# saving figure
print('saving results to logdir...')
fig_path = join(logdir, 'train_history.png')
title = 'Train History'
plt.title(title)
plt.savefig(fig_path)
print('all results saved.')

# printing execution message
print('training complete!')

# end of current module
