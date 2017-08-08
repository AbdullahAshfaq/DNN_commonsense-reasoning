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
data = pd.read_csv('dev.csv')
data_vec = pd.read_csv('dev_vec.csv')
# Dictionary for labels
#labels = {'neutral':0,'entailment':1,'contradiction':2}

#data = data.replace({'gold_label':labels})
print('............One Hot Encoding')
one_hot = pd.get_dummies(data['gold_label'],columns=['neutral','entailment','contradiction'])
print(one_hot.iloc[:10,1:])
'''
#Keras Model
model = Sequential()
model.add(Dense(100,input_dim=2400 , activation='relu',use_bias=True))
model.add(Dense(100, activation='relu',use_bias=True))
model.add(Dense(2400,activation='sigmoid'))
model.add()
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
log_likelihood = losses.log
model.compile(loss=,optimizer=sgd)
'''
#tensorflow model

n_nodes_hl1 = 3000
n_nodes_hl2 = 3000
n_nodes_hl3 = 2400
batch_size = 5
import math

def network(head,tail,r):

    weights = {
        'l1': tf.Variable(tf.truncated_normal(shape=[2403,n_nodes_hl1],stddev=1.0 / math.sqrt(float(n_nodes_hl1)))),
        'l2': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl1,n_nodes_hl2],stddev=1.0 / math.sqrt(float(n_nodes_hl2)))),
        'l3': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl2,n_nodes_hl3],stddev=1.0 / math.sqrt(float(n_nodes_hl3))))
    }

    biases = {
        'l1': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl1],stddev=1.0 / math.sqrt(float(n_nodes_hl1)))),
        'l2': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl2],stddev=1.0 / math.sqrt(float(n_nodes_hl2)))),
        'l3': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl3],stddev=1.0 / math.sqrt(float(n_nodes_hl3))))
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
    output = tf.nn.sigmoid(dot)#Output shape should be 5*1

    return output,dot

def loss_fun(y_true,y_pred):
    cost = tf.reduce_mean(-(tf.multiply(y_true, tf.log(y_pred)) + tf.multiply(1 - y_true, tf.log(1 - y_pred))))
    tf.summary.scalar('cost', cost)
    return cost

def train(y_true,y_pred, starter_learning_rate):
    #optimizer = tf.train.AdamOptimizer.minimize(loss)
    cost = tf.reduce_mean(-(tf.multiply(y_true, tf.log(y_pred)) + tf.multiply(1 - y_true, tf.log(1 - y_pred))))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               10, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(cost)
    return cost,train_op
'''
def train_network(head, tail, r, y_true):
    output = network(head,tail,r)
    log_likelihood = tf.reduce_mean(-(tf.multiply(y_true, tf.log(output)) + tf.multiply(1 - y_true, tf.log(1 - output))))
    optimizer = tf.train.AdamOptimizer().minimize(log_likelihood)
    epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, data.shape[0], batch_size):
                head = head_vec[i:i + batch_size]  # the result might be shorter than batchsize at the end
                tail = tail_vec[i:i + batch_size]
                labels = one_hot[i:i + batch_size]
                _,cost = sess.run([optimizer,log_likelihood],feed_dict={y_true})

'''
import numpy as np
import itertools

lr = 1e-4 #  Learning Rate
print('..............Starting graph')
with tf.Graph().as_default():
    #Inputs to the graph
    head_vec = tf.placeholder(dtype=tf.float32, shape=[None, 2400], name='head_vec')
    tail_vec = tf.placeholder(dtype=tf.float32, shape=[None, 2400], name='tail_vec')
    y_true_vec = tf.placeholder(dtype=tf.float32, shape=[None,1], name='y_true')
    rel_vec = tf.placeholder(dtype=tf.float32, shape=[None,3],name='relations')

    #relation = list(itertools.repeat([1, 0, 0], batch_size))  # Neutral relation
    relation = [[1,0,0],[0,1,0],[0,0,1]]
    #relation = np.random.rand(batch_size,50)
    #print(relation.shape)
    #relation = tf.Variable(tf.random_normal(shape=[None, 50]), name='relation')
    #Failed to convert object of type <class 'list'> to Tensor. Contents: [None, 50]. Consider casting elements to a supported type
    y_pred,dot = network(head_vec, tail_vec, rel_vec)
    #loss = loss(y_true=labels_vec, y_pred=y_pred)
    cost,train_op = train(y_true=y_true_vec, y_pred=y_pred, starter_learning_rate=lr)
    print_loss=[]
    epochs = 50
    print('..............Starting session')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            print('epoch = ', epoch)
            for i in range(0, 30, batch_size):
                head = data_vec.iloc[i:i+batch_size, 1:2401]
                tail = data_vec.iloc[i:i+batch_size, 2401:]
                labels = one_hot.iloc[i:i+batch_size,1:]
                labels = labels.values.tolist()
                #print(labels)
                #head = np.ones(shape=[batch_size,2400])
                #tail = np.ones(shape=[batch_size,2400])
                for rel in relation:
                    #print(rel)
                    y_true = np.int32([rel == l for l in labels])
                    y_true = np.expand_dims(y_true,axis=1)
                    #print(y_true.shape)
                    relation_vec = [items for items in itertools.repeat(rel, times=batch_size)]
                    #print(relation_vec)
                    l,t,prediction,d = sess.run([cost,train_op,y_pred,dot],feed_dict={head_vec:head, tail_vec:tail,
                                                                            y_true_vec:y_true,rel_vec:relation_vec})

                epoch_loss += l
                print('Batch = ',i,' to ', i+batch_size ,' loss = ',l)
                print('pred = ',prediction)
            print_loss.append(epoch_loss)
            print('epoch_loss = ',epoch_loss)

        head = data_vec.iloc[0:23, 1:2401]
        tail = data_vec.iloc[0:23, 2401:]
        labels = one_hot.iloc[0:23, 1:]
        relation_vec = [1,0,0]
        relation_vec = [items for items in itertools.repeat(relation_vec, 23)]
        prediction = sess.run([y_pred],feed_dict={head_vec:head, tail_vec:tail,rel_vec:relation_vec})
        print(prediction)


import matplotlib.pyplot as plt
plt.plot(print_loss)
plt.show()