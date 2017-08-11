from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras import losses
from tensorflow.python import debug as tf_debug


print('.......... Loading CSV files')
data = pd.read_csv('Train_reduced.csv')
data_vec = pd.read_csv('Train_vec_reduced.csv')
print('............One Hot Encoding')
one_hot = pd.get_dummies(data['gold_label'],columns=['neutral','entailment','contradiction'])
print(one_hot.iloc[:10,1:])
#tensorflow model

n_nodes_hl1 = 4800
n_nodes_hl2 = 4800
n_nodes_hl3 = 2400
batch_size = 5
import math

def DNN(head,tail,r):

    weights = {
        #'l1': tf.Variable(tf.truncated_normal(shape=[2403,n_nodes_hl1],stddev=1.0 / math.sqrt(float(n_nodes_hl1)))),
        #'l2': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl1,n_nodes_hl2],stddev=1.0 / math.sqrt(float(n_nodes_hl2)))),
        #'l3': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl2,n_nodes_hl3],stddev=1.0 / math.sqrt(float(n_nodes_hl3))))
        'l1': tf.get_variable("Wl1", shape=[2403,n_nodes_hl1],initializer=tf.contrib.layers.xavier_initializer()),
        'l2': tf.get_variable("Wl2", shape=[n_nodes_hl1, n_nodes_hl2], initializer=tf.contrib.layers.xavier_initializer()),
        'l3': tf.get_variable("Wl3", shape=[n_nodes_hl2, n_nodes_hl3], initializer=tf.contrib.layers.xavier_initializer())
    }

    biases = {
        #'l1': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl1],stddev=1.0 / math.sqrt(float(n_nodes_hl1)))),
        #'l2': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl2],stddev=1.0 / math.sqrt(float(n_nodes_hl2)))),
        #'l3': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl3],stddev=1.0 / math.sqrt(float(n_nodes_hl3))))
        'l1': tf.get_variable("bl1", shape=[n_nodes_hl1], initializer=tf.contrib.layers.xavier_initializer()),
        'l2': tf.get_variable("bl2", shape=[n_nodes_hl2], initializer=tf.contrib.layers.xavier_initializer()),
        'l3': tf.get_variable("bl3", shape=[n_nodes_hl3], initializer=tf.contrib.layers.xavier_initializer())
    }

    z0 = tf.concat([r,head],axis=1)

    l1 = tf.matmul(z0,weights['l1']) + biases['l1']
    z1 = tf.nn.relu(l1)

    l2 = tf.matmul(z1,weights['l2']) + biases['l2']
    z2 = tf.nn.relu(l2)

    l3 = tf.matmul(z2,weights['l3']) + biases['l3']
    z3 = tf.nn.relu(l3)

    m = tf.multiply(z3,tail)
    dot = tf.reduce_sum(m,axis=1)
    output = tf.nn.sigmoid(dot)#Output shape should be batch_size*1

    return output,dot

def train(y_true,y_pred, starter_learning_rate):
    cost = tf.reduce_mean(-(tf.multiply(y_true, tf.log(y_pred)) + tf.multiply(1 - y_true, tf.log(1 - y_pred))))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,decay_steps=5,
                                               decay_rate=0.5, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return cost,train_op,learning_rate

import numpy as np
import itertools

lr = 1e-3 #  Learning Rate
print('..............Starting graph')
with tf.Graph().as_default():
    #Inputs to the graph
    head_vec = tf.placeholder(dtype=tf.float32, shape=[None, 2400], name='head_vec')
    tail_vec = tf.placeholder(dtype=tf.float32, shape=[None, 2400], name='tail_vec')
    y_true_vec = tf.placeholder(dtype=tf.float32, shape=[None,1], name='y_true')
    rel_vec = tf.placeholder(dtype=tf.float32, shape=[None,3],name='relations')

    relation = [0,1,0] # This is the relation code for "entailment"
    
    y_pred,dot = DNN(head_vec, tail_vec, rel_vec)
    
    cost,train_op,lear = train(y_true=y_true_vec, y_pred=y_pred, starter_learning_rate=lr)
    print_loss=[]
    learning = []
    epochs = 20
    print('..............Starting session')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            print('epoch = ', epoch)
            for i in range(0, 30, batch_size):#For now, Just train on first 30 examples
                head = data_vec.iloc[i:i+batch_size, 1:2401]# head vec from index 1 to 2400
                tail = data_vec.iloc[i:i+batch_size, 2401:]# tail vec from index 2401 to 4800
                labels = one_hot.iloc[i:i+batch_size,1:]# labels are one hot encoded
                labels = labels.values.tolist()
                #print(labels)
                #head = np.ones(shape=[batch_size,2400])
                #tail = np.ones(shape=[batch_size,2400])
                    #print(rel)
                y_true = np.int32([relation == l for l in labels])# If labels are equal to relation, then it will belong to +ive class
                y_true = np.expand_dims(y_true,axis=1)
                #print(y_true.shape)
                relation_vec = [items for items in itertools.repeat([0,0,0], times=batch_size)]# for now, feed zeros into relation vector
                #print(relation_vec)
                l,t,prediction,d,le = sess.run([cost,train_op,y_pred,dot,lear],feed_dict={head_vec:head, tail_vec:tail,
                                                                        y_true_vec:y_true,rel_vec:relation_vec})

                epoch_loss += l
                learning.append(le)
                print('Batch = ',i,' to ', i+batch_size ,' loss = ',l)
                print('pred = ',prediction)
            print_loss.append(epoch_loss)
            print('epoch_loss = ',epoch_loss)
	# Check probability of first 10 being entailment
        head = data_vec.iloc[0:10, 1:2401]
        tail = data_vec.iloc[0:10, 2401:]
        labels = one_hot.iloc[0:10, 1:]
        relation_vec = [0,0,0]
        relation_vec = [items for items in itertools.repeat(relation_vec, 10)]
        prediction = sess.run([y_pred],feed_dict={head_vec:head, tail_vec:tail,rel_vec:relation_vec})
        print('testing = ',prediction)

# Plotting loss function
import matplotlib.pyplot as plt
plt.plot(print_loss)
plt.show()
