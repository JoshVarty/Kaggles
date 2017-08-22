import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import pandas as pd
from random import randint

#Load data  
data = pd.read_csv("../input/train.csv");

#Partition into train/test sets
images = data.iloc[0:5000,1:]
labels = data.iloc[0:5000,:1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

#Sanity check a random image
i= randint(0, len(train_images))
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.figure()
plt.imshow(img)
plt.title(train_labels.iloc[i,0])
plt.show();

#Cleanup
del data
del images
del labels

#Converto numpy arrays for tensorflow
train_images = train_images.as_matrix()
test_images = test_images.as_matrix()
train_labels = train_labels.as_matrix()
test_labels = test_labels.as_matrix()



