# for loading/processing the images
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input

from src.utils.aux_funcs import is_using_gpu

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

path = '/mnt/sdb/angelo-dados/pycharm_projects/single_cell_analyser/data/clustering_tests/flowers_dataset/flower_images/flower_images'
# change the working directory to the path where the images are located
os.chdir(path)

print('Using GPU:', is_using_gpu())

# this list holds all the image filename
flowers = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
    # loops through each file in the directory
    for file in files:
        if file.name.endswith('.png'):
            # adds only the image files to the flowers list
            flowers.append(file.name)

f_string = f'{len(flowers)} images found'
print(f_string)

model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)


def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx,
                             use_multiprocessing=True)
    return features


data = {}
p = '/mnt/sdb/angelo-dados/pycharm_projects/single_cell_analyser/data/clustering_tests/flowers_dataset/flower_images/flower_images/features.pickle'

# loop through each image in the dataset
for flower in flowers:
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(flower, model)
        data[flower] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        with open(p, 'wb') as file:
            pickle.dump(data, file)

# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))
print(feat)
exit()

# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1, 4096)

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=2,
          random_state=53)  # seed
pca.fit(feat)
x = pca.transform(feat)

# cluster feature vectors
kmeans = KMeans(n_clusters=15,  # if we know how many groups there are
                random_state=53,  # seed
                n_init=10)  # kind of a bootstrap
kmeans.fit(x)

# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

labels = kmeans.labels_

label0 = x[labels == 0]
label0x = label0[:, 0]
label0y = label0[:, 1]
print(label0)
print(label0x)
print(label0y)
print(x.shape)

# Getting unique labels

u_labels = np.unique(labels)

# plotting the results:

for i in u_labels:
    plt.scatter(x[labels == i, 0], x[labels == i, 1], label=i)
plt.legend()
plt.show()
plt.close()


# function that lets you view a cluster (based on identifier)
def view_cluster(cluster):
    plt.figure(figsize=(25, 25))
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10, 10, index + 1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


view_cluster(8)

# this is just incase you want to see which value for k might be the best
sse = []
list_k = list(range(3, 51))

for k in list_k:
    km = KMeans(n_clusters=k,
                n_init=10,
                random_state=53)
    km.fit(x)

    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
plt.show()

# end of current module
