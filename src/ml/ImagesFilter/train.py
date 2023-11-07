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
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.metrics import Recall
from src.utils.aux_funcs import IMAGE_WIDTH
from src.utils.aux_funcs import IMAGE_HEIGHT
from tensorflow.keras.metrics import Precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.utils import image_dataset_from_directory
print('all required libraries successfully imported.')  # noqa

######################################################################
# defining global variables

# TODO: add argsparser and main

data_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\imgs\\dataset'
# data_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\imgs\\dataset'
logdir = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\logs'
# logdir = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\logs'
save_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\models\\modelV1.h5'
# save_path = 'Z:\\pycharm_projects\\single_cell_analyser\\data\\nucleus_detection\\ImagesFilter\\models\\modelV1.h5'
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
epochs = 1

######################################################################
# running training

# loading data
print(f'loading data from folder "{data_path}"...')
data = image_dataset_from_directory(directory=data_path,
                                    labels='inferred',
                                    label_mode='binary',
                                    class_names=['excluded', 'included'],
                                    color_mode='rgb',
                                    batch_size=8,
                                    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                    shuffle=True)

# normalizing data to 0-1 scale
print('normalizing data...')
data = data.map(lambda x, y: (x / 255, y))
data_len = len(data)

# getting split sizes
train_size = int(data_len * train_ratio)
val_size = int(data_len * val_ratio)
test_size = int(data_len * test_ratio)

# getting data splits
print('getting data splits...')
# TODO: change this to structure in folder and grab them by the same data taker function
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
f_string = f'Train: {train_ratio * 100}%\n'
f_string += f'Val: {val_ratio * 100}%\n'
f_string += f'Test: {test_ratio * 100}%'
print(f_string)

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
print('training complete!')

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

# testing model on test split
print('testing model on test split...')

# starting precision/recall/accuracy instances
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()

# getting test batches
test_batches = test.as_numpy_iterator()

# iterating over batches in test data set
for batch in test_batches:
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)

# getting results
precision_result = precision.result()
recall_result = recall.result()
accuracy_result = accuracy.result()

# printing results
print('Precision: ', precision_result)
print('Recall: ', recall_result)
print('Accuracy: ', accuracy_result)

# printing execution message
print('done!')

# end of current module
