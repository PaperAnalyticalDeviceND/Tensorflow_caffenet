# Load libraries. We need tensorflow and numpy for training the CNN,
# matplotlib for plotting, os.path and PIL for loading images,
# and random to partition the data into test and training sets
import tensorflow as tf
import numpy as np
#from matplotlib import pyplot as plt
import math
from os.path import dirname, abspath
from os.path import join
import os.path
import PIL
import random
#import urllib.request
from PIL import Image, ImageEnhance, ImageStat
import sys
import getopt

#get inlines
#inline parameter?
optlist, args = getopt.getopt(sys.argv[1:], 'i:')

#image to analyzed
image_location = '/var/www/html/joomla/images/padimages/processed//Acetaminophen-12LanePADKenya2015-1-58861.processed.png'

for o, a in optlist:
    if o == '-i':
        image_location = a
        print("Image location", image_location)
    else:
        print('Unhandled option: ', o)
        sys.exit(-2)

#variables
classes = 9 #Amoxicillin rerun,Acetaminophen,Ciprofloxacin,Ceftriaxone,Metformin,Ampicillin,Azithromycin,Cefuroxime Axetil,Levofloxacin

# Parameters for training
#learning rate is the step size that we move in the opposite direction of the gradient for each weight/bias
learning_rate = 1e-4
#an epoch represents having trained with a number of images equal to the training test size
max_epochs = 100
display_step_size = 10 # Number of iterations before checking on the performance of network (validation)
#the prob_keep parameter represents the dropout, probability to keep units. This makes the network more robust
prob_keep = 0.5

#max pooling values
mx_pooling_size = 2        #max pooling size 2 *2
mx_pooling_window = 3      #max pooling window 3 *3

# Convolutional Layer 1.
filter_size1 = 11          # Convolution filters are 11 x 11 pixels.
num_filters1 = 96
filter_stride1 = 4

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 256

# Convolutional Layer 3.
filter_size3 = 3          # Convolution filters are 3 x 3 pixels.
num_filters3 = 384

# Convolutional Layer 4.
filter_size4 = 3          # Convolution filters are 3 x 3 pixels.
num_filters4 = 384

# Convolutional Layer 5.
filter_size5 = 3          # Convolution filters are 3 x 3 pixels.
num_filters5 = 256

# Fully-connected layers.
fc_neurons1 = 4096         # Number of neurons in fully-connected layer.
fc_neurons2 = 4096         # Number of neurons in second fully-connected layer.


#Image properties
img_shape = (227, 227)

#the images are color so 3 channels
channels = 3

#location of the weights/biases and architecture for the CNN checkpoint
model_checkpoint = "tmp/caffenet_pad_1.ckpt"

#image batch size
image_batch_size = 128

# Define CNN
#create place holders, these will be loaded with data as we train
X = tf.placeholder(tf.float32, shape=[None, img_shape[0]*img_shape[1]*channels], name='X') #the set of input images
X_image = tf.reshape(X, [-1, img_shape[0], img_shape[1], channels]) #a single image
Y = tf.placeholder(tf.float32, shape=[None, classes], name='Y')     #the label set
Y_classes = tf.argmax(Y, axis=1)                               #the classes(subjects)

#dropout (keep probability)
keep_prob = tf.placeholder(tf.float32)

#initial weight values
def generate_weights(shape, name):
    # Create new matrix
    return tf.Variable(tf.truncated_normal(shape, stddev=5e-2), name=name)

#initial bias values
def generate_biases(size, name):
    #create biases
    return tf.Variable(tf.constant(0.0, shape=[size]), name=name)

# compute convolutions with relu output
def convolution(input_data,num_channels, filter_size, num_filters, stride, name_w, name_b):
    #shape for weights
    shape = [filter_size, filter_size, num_channels, num_filters]
    # Generate new weights
    W = generate_weights(shape=shape, name=name_w)
    # generate new biases, one for each filter.
    b= generate_biases(size=num_filters, name=name_b)
    #tensorflow convolution
    out = tf.nn.conv2d(input=input_data, filter=W, strides=[1, stride, stride, 1], padding='SAME')
    # Add the biases
    out= tf.nn.bias_add(out,b)
    #relu activation
    out = tf.nn.relu(out)
    return out, W, b

#max pooling layer
def max_pooling(input_data,size,window):
    out = tf.nn.max_pool(value=input_data, ksize=[1, window, window, 1], strides=[1, size, size, 1], padding='SAME')
    return out

def reduce(tensor):
    #reduce the 4-dim tensor, the output from the
    #conv/maxpooling to 2-dim as input to the fully-connected layer
    features = tensor.get_shape()[1:4].num_elements() # The volume
    reduced = tf.reshape(tensor, [-1, features])
    return reduced, features

#compute the fully connected layer
def compute_fc_layer(input_data,input_size, output_size, name_w, name_b, use_relu=True, user_dropout=False):
    # generate new weights and biases.
    W = generate_weights(shape=[input_size, output_size], name=name_w)
    b = generate_biases(size=output_size, name=name_b)
    #compute the out
    out = tf.matmul(input_data, W) + b
    # Use ReLU?
    if use_relu:
        out = tf.nn.relu(out)
    #Add dropout regularisation if its not the out layer
    if user_dropout:
         out = tf.nn.dropout(out, keep_prob)
    return out, W, b

#CNN architecture
#layer 1
conv_layer1, conv_W1, conv_B1 = convolution(input_data=X_image,num_channels=channels,
                                   filter_size=filter_size1,num_filters=num_filters1, stride=filter_stride1, name_w='conv_W1', name_b='conv_B1')

#layer 1 has max pooling
max_pooling_layer_1 = max_pooling(input_data=conv_layer1,size=mx_pooling_size,window=mx_pooling_window)

#need to normalize here
#norm1 = tf.nn.lrn(max_pooling_layer_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

#Layer 2
conv_layer2, conv_W2, conv_B2 =convolution(input_data=max_pooling_layer_1,num_channels=num_filters1,
                                  filter_size=filter_size2,num_filters=num_filters2, stride=1, name_w='conv_W2', name_b='conv_B2')

#layer 2 has max pooling
max_pooling_layer_2 = max_pooling(input_data=conv_layer2,size=mx_pooling_size,window=mx_pooling_window)

#need to normalize here
#norm2 = tf.nn.lrn(max_pooling_layer_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

#Layer 3
conv_layer3, conv_W3, conv_B3 =convolution(input_data=max_pooling_layer_2,num_channels=num_filters2,
                                  filter_size=filter_size3,num_filters=num_filters3, stride=1, name_w='conv_W3', name_b='conv_B3')

#Layer 4
conv_layer4, conv_W4, conv_B4 =convolution(input_data=conv_layer3,num_channels=num_filters3,
                                  filter_size=filter_size4,num_filters=num_filters4, stride=1, name_w='conv_W4', name_b='conv_B4')

#Layer 5
conv_layer5, conv_W5, conv_B5 =convolution(input_data=conv_layer4,num_channels=num_filters4,
                                  filter_size=filter_size5,num_filters=num_filters5, stride=1, name_w='conv_W5', name_b='conv_B5')

#layer 2 has max pooling
max_pooling_layer_5 = max_pooling(input_data=conv_layer5,size=mx_pooling_size,window=mx_pooling_window)

#reshape the out from covolution layers for input into fully connected layers
Xi, features = reduce(max_pooling_layer_5)

#Fully connected layers
FC1, fc_W1, fc_B1 = compute_fc_layer(input_data=Xi,input_size=features,
                       output_size=fc_neurons1, name_w='fc_W1', name_b='fc_B1', use_relu=True,user_dropout=True)

FC2, fc_W2, fc_B2 = compute_fc_layer(input_data=FC1,input_size=fc_neurons1,
                       output_size=fc_neurons2, name_w='fc_W2', name_b='fc_B2', use_relu=True, user_dropout=True)

FC3, fc_W3, fc_B3 = compute_fc_layer(input_data=FC2,input_size=fc_neurons2,
                       output_size=classes, name_w='fc_W3', name_b='fc_B3', use_relu=False, user_dropout=False)

output = tf.nn.softmax(FC3) #softmax output
pred = tf.argmax(output, axis=1) # predictions
#compute cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=FC3, labels=tf.stop_gradient(Y)))
#optimise the cost function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#Compute Accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, Y_classes), tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver({'conv_W1': conv_W1,'conv_B1': conv_B1,'conv_W2': conv_W2,'conv_B2': conv_B2,'conv_W3': conv_W3,'conv_B3': conv_B3,'conv_W4': conv_W4,'conv_B4': conv_B4,'conv_W5': conv_W5,'conv_B5': conv_B5,'fc_W1': fc_W1,'fc_B1': fc_B1,'fc_W2': fc_W2,'fc_B2': fc_B2,'fc_W3': fc_W3,'fc_B3': fc_B3}) #tf.global_variables ())

#location of the weights/biases and architecture for the CNN (saved above)
model_for_identification = model_checkpoint

#called to identify subject in image
def identify(image):
    #reshape the image to an (1, 154587) vector
    image = np.mat(np.asarray(image).flatten())
    #create session
    with tf.Session() as sess:
        #load in the saved weights
        saver.restore(sess, model_for_identification)
        #find the prediction
        result = sess.run(pred, feed_dict={X: image, keep_prob: 1.})
        #we can look at the softmax output as well
        prob_array = sess.run(output, feed_dict={X: image, keep_prob: 1.})
        #print("op",prob_array)

        #return result (add 1 as indexed from 1 not 0)
        return result[0], prob_array

#load drug names
f = open('drug_names.txt')
firstline = f.readline()
drugs = firstline.split(',')

#get it
#urllib.request.urlretrieve( 'https://pad.crc.nd.edu/images/padimages/processed//Acetaminophen-12LanePADKenya2015-1-58861.processed.png', 'test.jpg')
#Load png file using the PIL library
img = PIL.Image.open(image_location)
#crop out active area
img = img.crop((71, 359, 71+636, 359+490))
#lanes split
lane = []

#exclude these lanes
exclude = "AJ"

#loop over lanes
for i in range(0,12):
    if chr(65+i) not in exclude:
        lane.append(img.crop((53*i, 0, 53*(i+1), 490)))

#reconstruct
imgout = Image.new("RGB", (53 * len(lane), 490))

#loop over lanes
for i in range(0,len(lane)):
    imgout.paste(lane[i], (53*i, 0, 53*(i+1), 490))

#resize
imgout = imgout.resize((227,227), Image.ANTIALIAS)

#show it
#imgplot = plt.imshow(imgout)

#catagorize
predicted_drug, predicted_prob = identify(imgout)

print("Drug predicted to be", drugs[predicted_drug])
