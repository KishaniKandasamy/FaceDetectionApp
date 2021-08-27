import pandas as pd
import random
import numpy as np
import os
import PIL
import seaborn as sns
import pickle
from PIL import *
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.python.keras import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, optimizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

keyfacial_df = pd.read_csv('./../data.csv')
keyfacial_df.info()

# Check if null values exist in the dataframe
keyfacial_df.isnull().sum()

#initially 1D array
keyfacial_df['Image'].shape

#Image are given as space separated string, separate the values using ' ' as separator. 1D->2D
keyfacial_df['Image'] = keyfacial_df['Image'].apply(lambda x: np.fromstring(x, dtype = int, sep = ' ').reshape(96, 96))
keyfacial_df['Image'][0].shape

#particular column details
keyfacial_df['right_eye_center_x'].describe()

#x-coordinates are in even columns like 0,2,4,.. and y-coordinates are in odd columns like 1,3,5,..
i = np.random.randint(1, len(keyfacial_df)) # as o th row is title
plt.imshow(keyfacial_df['Image'][i], cmap = 'gray')
for j in range(1, 31, 2):
        plt.plot(keyfacial_df.loc[i][j-1], keyfacial_df.loc[i][j], 'rx')
    
# Let's view more images in a grid format
fig = plt.figure(figsize=(20, 20))

for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1) 
    x = np.random.randint(1, len(keyfacial_df))
    image = plt.imshow(keyfacial_df['Image'][x],cmap = 'gray')
    for j in range(1,31,2):
        plt.plot(keyfacial_df.loc[x][j-1], keyfacial_df.loc[x][j], 'rx')
        
#Data Augmentation
# Create a new copy of the dataframe
import copy
keyfacial_df_copy = copy.copy(keyfacial_df)

# Obtain the columns in the dataframe
columns = keyfacial_df_copy.columns[:-1]
columns

# Horizontal flip the images along y axis(1)
keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x: np.flip(x, axis = 1))

# horizontal- y coordinate values would be the same
# Only x coordiante values would change( width of the image(96) - x) & only x cordinate only even numbers
for i in range(len(columns)):
    if i%2 == 0:
        keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(lambda x: 96. - float(x) )
        
#  Original image
plt.imshow(keyfacial_df['Image'][9], cmap = 'gray')
for j in range(1, 31, 2):
        plt.plot(keyfacial_df.loc[9][j-1], keyfacial_df.loc[9][j], 'rx')
        
#  Horizontally flipped image
plt.imshow(keyfacial_df_copy['Image'][9],cmap='gray')
for j in range(1, 31, 2):
        plt.plot(keyfacial_df_copy.loc[9][j-1], keyfacial_df_copy.loc[9][j], 'rx')
        
# Concatenate  the flipped image
augmented_df = np.concatenate((keyfacial_df, keyfacial_df_copy))
augmented_df.shape        
