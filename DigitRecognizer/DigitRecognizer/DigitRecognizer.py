import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from random import randint
import numpy as np

image_size = 28     #28x28 images
num_channels = 1    #Not RGB, just single values for each pixel
num_labels = 10     #0-9

batch_size = 128
patch_size = 5
depth = 16
num_hidden = 64


#Load data  
data = pd.read_csv("../input/train.csv");
test_data = pd.read_csv("../input/test.csv");
test_data = test_data.as_matrix().reshape((-1, image_size, image_size, num_channels)).astype(np.float32)


#Partition into train/test sets
images = data.iloc[0:5000,1:]
labels = data.iloc[0:5000,:1]
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

#Cleanup
del data
del images
del labels


def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  labels = np.reshape(labels, (len(labels), num_labels))
  return dataset, labels

print("Shapes Before Reformat")
print("train_images: ", train_images.shape)
print("train_labels: ", train_labels.shape)
print("test_images: ", test_images.shape)
print("test_labels: ", test_labels.shape)

#Convert to numpy arrays for tensorflow
train_images, train_labels = reformat(train_images.as_matrix(), train_labels.as_matrix())
test_images, test_labels = reformat(test_images.as_matrix(), test_labels.as_matrix())



def ConvNet():

    graph = tf.Graph()
    with graph.as_default():

      # Input data.
      tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_test_dataset = tf.constant(test_images, dtype=tf.float32)
      
      # Variables.
      layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
      layer1_biases = tf.Variable(tf.zeros([depth]))
      layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
      layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
      layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
      layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
      layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
      layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

      # Model.
      def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        shape = pool_1.get_shape().as_list()
        reshape = tf.reshape(pool_1, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

      def accuracy(predictions, labels):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
      
      # Training computation.
      logits = model(tf_train_dataset)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
        
      # Optimizer.
      optimizer = tf.train.AdamOptimizer(0.000005).minimize(loss)
      
      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      test_prediction = tf.nn.softmax(model(tf_test_dataset))

      num_steps = 15001

    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_images[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))

      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
      
      #Save session
      saver = tf.train.Saver()
      save_path = saver.save(session, "model/model.ckpt")



def LoadAndRun():
    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():
      # Input data.
      tf_test_dataset = tf.constant(test_data, dtype=tf.float32)
      
      # Variables.
      layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
      layer1_biases = tf.Variable(tf.zeros([depth]))
      layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
      layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
      layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
      layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
      layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
      layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

      # Model.
      def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv = tf.nn.conv2d(pool_1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        pool_1 = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        shape = pool_1.get_shape().as_list()
        reshape = tf.reshape(pool_1, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

      test_prediction = tf.nn.softmax(model(tf_test_dataset))


      with tf.Session(graph=graph) as session:
        saver = tf.train.Saver()
        saver.restore(session, "model/model.ckpt")
        print("Restored")

        x = test_prediction.eval();
        results = np.argmax(x, 1)
        print(results)


LoadAndRun();