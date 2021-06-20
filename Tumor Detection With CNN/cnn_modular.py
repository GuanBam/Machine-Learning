import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

###########################################################
#parameters for the main loop

learning_rate = 0.005
n_epochs = 270000
batch_size = 256
display_step = 10


# Parameters for the network

n_input = 10000 # Feature for pixeals 100*100
n_classes = 2 # Yes and No
dropout = 0.7 # Dropout, probability to keep units


# load data for training and test
x_train=np.genfromtxt("data.csv",delimiter=',',skip_header=1,usecols=(i for i in range(1,100*100+1)))
y_train=np.genfromtxt("data.csv",delimiter=',',skip_header=1,usecols=0)

x_test=np.genfromtxt("data1.csv",delimiter=',',skip_header=1,usecols=(i for i in range(1,100*100+1)))
y_test=np.genfromtxt("data1.csv",delimiter=',',skip_header=1,usecols=0)

############################################################
## normalizing

sc = StandardScaler()
sc.fit(x_train)

x_train_normalized = sc.transform(x_train)
x_test_normalized  = sc.transform(x_test)


def convertOneHot_data2(data):
    y=np.array([int(i) for i in data])
    rows = len(y)
    columns = y.max() + 1 - y.min()
    tmp = y.min()
    a = np.zeros(shape=(rows,columns))
    print "onehot rows:",rows
    print "onehot colums:",columns
    for i,j in enumerate(y):
        a[i][j-tmp]=1
    return (a)

y_train_onehot = convertOneHot_data2(y_train)
y_test_onehot  = convertOneHot_data2(y_test)

###################################################################
## print stats 
precision_scores_list = []
accuracy_scores_list = []

def print_stats_metrics(y_test, y_pred):    
    print('Accuracy: %.2f' % accuracy_score(y_test,   y_pred) )
    accuracy_scores_list.append(accuracy_score(y_test,   y_pred) )
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print "confusion matrix"
    print(confmat)
    print pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    precision_scores_list.append(precision_score(y_true=y_test, y_pred=y_pred,average='weighted'))
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred,average='weighted'))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred,average='weighted'))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred,average='weighted'))

#####################################################################

def plot_metric_per_epoch():
    x_epochs = []
    y_epochs = [] 
    for i, val in enumerate(accuracy_scores_list):
        x_epochs.append(i)
        y_epochs.append(val)
    
    plt.scatter(x_epochs, y_epochs,s=50,c='lightgreen', marker='s', label='score')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.title('Score per epoch')
    plt.legend()
    plt.grid()
    plt.show()

########################################################################

def conv2d(x, W, b, strides=1):
    # Conv2D function, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


##########################################################################

def maxpool2d(x, k=2):
    # MaxPool2D function
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


################################################################

def layer(input, weight_shape, bias_shape):
    W = tf.Variable(tf.truncated_normal(weight_shape,stddev=0.1))
    b = tf.Variable(tf.constant(0.1,shape=bias_shape))
    mapping = tf.matmul(input, W)   
    result = tf.add( mapping ,  b )
    return tf.nn.softmax(result)


################################################################

def conv_layer(input, weight_shape, bias_shape):
    ##rr =raw_input()
    W = tf.Variable(tf.truncated_normal(weight_shape,stddev=0.1))
    b = tf.Variable(tf.constant(0.1,shape=bias_shape))
    conv = conv2d(input, W, b)
    # Max Pooling (down-sampling)
    conv_max = maxpool2d(conv, k=2)
    return conv_max

################################################################

def fully_connected_layer(conv_input, fc_weight_shape, fc_bias_shape, dropout):   
    new_shape = [-1, tf.Variable(tf.random_normal(fc_weight_shape)).get_shape().as_list()[0]]
    fc = tf.reshape(conv_input, new_shape)
    mapping = tf.matmul(fc, tf.Variable(tf.random_normal( fc_weight_shape))   )
    fc = tf.add( mapping, tf.Variable(tf.random_normal(fc_bias_shape))    )
    fc = tf.nn.relu(fc)
    # Apply Dropout
    fc = tf.nn.dropout(fc, dropout)
    return fc


###########################################################
## define the architecture here

def inference_conv_net2(x, dropout):
    # Reshape input picture 
    # shape = [-1, size_image_x, size_image_y, 1 channel (e.g. grey scale)]
    x = tf.reshape(x, shape=[-1, 100, 100, 1])

    # Convolution Layer 1, filter 10x10 conv, 1 input, 16 outputs
    # max pool will reduce image from 100*100 to 50*50
    conv1 = conv_layer(x, [10, 10, 1, 16], [16] )
    
    # Convolution Layer 2, filter 10x10 conv, 16 inputs, 36 outputs
    # max pool will reduce image from 50*50 to 25*25
    conv2 = conv_layer(conv1, [10, 10, 16, 32], [32] )
    
    # Fully connected layer, 25*25*32 inputs, 64 outputs
    # Reshape conv2 output to fit fully connected layer input
    fc1 = fully_connected_layer(conv2, [25*25*32, 64], [64] , dropout)
    
    # Output, 64 inputs, 2 outputs (class prediction)
    output = layer(fc1 ,[64, n_classes], [n_classes] )
    return output

###########################################################

def loss_deep_conv_net(output, y_tf):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y_tf)
    loss = tf.reduce_mean(xentropy) 
    return loss


###########################################################

def training(cost):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost)
    return train_op


###########################################################
 

def evaluate(output, y_tf):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_tf,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


###########################################################


x_tf = tf.placeholder(tf.float32, [None, n_input])
y_tf = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


###############################################################
         
output = inference_conv_net2(x_tf, keep_prob) 
cost = loss_deep_conv_net(output, y_tf)

train_op = training(cost) 
eval_op = evaluate(output, y_tf)


##################################################################
## for metrics

y_p_metrics = tf.argmax(output, 1)

##################################################################
# Initialize and run

init = tf.global_variables_initializer() 
sess = tf.Session()
sess.run(init)

y_test_temp = y_test_onehot
y_train_temp = y_train_onehot

###########################################################################################
dropout2 = 1.0

num_samples_train =  len(y_train)
print num_samples_train
num_batches = int(num_samples_train/batch_size)


# Keep training until reach max iterations
print "running..."
for i in range(n_epochs):
    average_accuracy=0
    for batch_n in range(num_batches):
    	sta=batch_n*batch_size
    	end=sta+batch_size
    	sess.run(train_op, feed_dict={x_tf: x_train_normalized[sta:end,:], y_tf: y_train_temp[sta:end, :], keep_prob: dropout})

   	result, y_result_metrics = sess.run([eval_op, y_p_metrics], feed_dict={x_tf: x_test_normalized,y_tf: y_test_temp,keep_prob: dropout2})
        
    print "test {},{}".format(i,result)
    y_true = np.argmax(y_test_temp,1)
    print_stats_metrics(y_true, y_result_metrics)
    if i == 1000:
        plot_metric_per_epoch()



##########################################################################################

print "<<<<<<DONE>>>>>>"





