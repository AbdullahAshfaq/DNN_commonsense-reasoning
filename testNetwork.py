import tensorflow as tf
import itertools
import numpy as np
import pandas as pd
tf.logging.set_verbosity(tf.logging.INFO)

n_nodes_hl1 = 50
n_nodes_hl2 = 3
n_nodes_hl3 = 1
batch_size = 20

import math

def DNN(head,tail):

    weights = {
        'l1': tf.Variable(tf.truncated_normal(shape=[4,n_nodes_hl1],stddev=1.0 / math.sqrt(float(n_nodes_hl2)),dtype=tf.float64)),
        'l2': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl1,n_nodes_hl2],stddev=1.0 / math.sqrt(float(n_nodes_hl2)),dtype=tf.float64)),
        #'l3': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl2,n_nodes_hl3],stddev=1.0 / math.sqrt(float(n_nodes_hl3))))
        #'l1': tf.get_variable("Wl1", shape=[2,n_nodes_hl1],initializer=tf.contrib.layers.xavier_initializer()),
        #'l2': tf.get_variable("Wl2", shape=[n_nodes_hl1, n_nodes_hl2], initializer=tf.contrib.layers.xavier_initializer()),
        #'l3': tf.get_variable("Wl3", shape=[n_nodes_hl2, n_nodes_hl3], initializer=tf.contrib.layers.xavier_initializer())
    }

    biases = {
        'l1': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl1],stddev=1.0 / math.sqrt(float(n_nodes_hl2)),dtype=tf.float64)),
        'l2': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl2],stddev=1.0 / math.sqrt(float(n_nodes_hl2)),dtype=tf.float64)),
        #'l3': tf.Variable(tf.truncated_normal(shape=[n_nodes_hl3],stddev=1.0 / math.sqrt(float(n_nodes_hl3))))
        #'l1': tf.get_variable("bl1", shape=[n_nodes_hl1], initializer=tf.contrib.layers.xavier_initializer()),
        #'l2': tf.get_variable("bl2", shape=[n_nodes_hl2], initializer=tf.contrib.layers.xavier_initializer()),
        #'l3': tf.get_variable("bl3", shape=[n_nodes_hl3], initializer=tf.contrib.layers.xavier_initializer())
    }
    #relation = tf.Variable(tf.random_normal(shape=[batch_size,1],dtype=tf.float64))
    relation = tf.constant([0.,0.,0.,0.,0.],dtype=tf.float64,shape=[batch_size,1])
    #print(weights['l1'])
    z0 = tf.concat([head,relation],axis=1)
    #print(z0)
    l1 = tf.matmul(z0,weights['l1'])+ biases['l1']
    z1 = tf.nn.relu(l1)

    l2 = tf.matmul(z1,weights['l2'])+ biases['l2']
    #l2 = tf.nn.relu(l2)
    print(l2,tail)
    #################################################
    # DO NOT USE RELU BEFORE SIGMOID
    #################################################
    m = tf.multiply(tail,l2)
    m = tf.reduce_mean(m,axis=1)
    print('m size = ',m)
    z2 = tf.nn.sigmoid(m)
    '''
    l3 = tf.matmul(z2,weights['l3'])+ biases['l3']
    z3 = tf.nn.relu(l3)
    #m = tf.multiply(z3,tail)
    print(z3)
    #dot = tf.reduce_sum(z3,axis=1)
    '''
    #output = tf.nn.sigmoid(z1)#Output shape should be 5*1
    #z2 = tf.reduce_sum(z2)
    return z2,relation

'''
def make_pos_Input():
    H = []
    for i in range(10):
        H.append([items for items in itertools.repeat(i+1, times=10)])
    H = np.asarray(H)
    T = -H
    labels = [items for items in itertools.repeat(1, times=10)]
    return H,T,labels

def make_neg_Input(head,tail):
    #H = np.array([])
    H = np.array([items for items in itertools.repeat([items for items in itertools.repeat(1, times=10)], times=9)])
    T = []
    for i in range(9):
        temp = np.array([items for items in itertools.repeat([items for items in itertools.repeat(i + 2, times=10)], times=9)])
        H = np.concatenate([H,temp],axis=0)
    #print(tail)
    count = 0
    for h in head:
        #H.append([items for items in itertools.repeat(h, times=9)])
        for t in tail:
            if (h == -t).all():
                continue
            T.append(t)

    H = np.asarray(H)
    #print('Hshape = ',H)
    #T.pop()
    T = np.asarray(T)
    labels = np.array([items for items in itertools.repeat(0, times=H.shape[0])])
    np.expand_dims(labels,axis=1)
    return H,T,labels

H,T,labels = make_pos_Input()
Hn,Tn,labelsn = make_neg_Input(H,T)
Tdataset = np.column_stack([H,T,labels])
print(Tdataset.shape)
Fdataset = np.column_stack([Hn,Tn,labelsn])
print(Fdataset.shape)
final = np.concatenate([Tdataset,Fdataset],axis=0)

from sklearn.utils import shuffle
final = shuffle(final)

final = pd.DataFrame(final)


final.to_csv('dummy_data.csv')
print(final.shape)
'''
def train(y_true,y_pred, starter_learning_rate):
    #optimizer = tf.train.AdamOptimizer.minimize(loss)
    cost = tf.reduce_mean((tf.multiply(-y_true, tf.log(y_pred+1e-10)) + tf.multiply(-1 + y_true, tf.log(1 - y_pred + 1e-10))),axis=1)
    print('cost shape = ', cost)
    cost = tf.reduce_mean(cost)
    print('cost shape = ',cost)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,decay_steps=10,
                                               decay_rate=0.2, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #gvs = optimizer.compute_gradients(cost)
    #capped_gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
    #global_step = tf.Variable(0, name='global_step', trainable=False)
    #train_op = optimizer.apply_gradients(capped_gvs,global_step=global_step)
    train_op = optimizer.minimize(cost,global_step=global_step)
    return cost,train_op,learning_rate


data_vec = pd.read_csv('dum.csv')

lr = 1e-3 #  Learning Rate
print('..............Starting graph')
with tf.Graph().as_default():
    head_vec = tf.placeholder(dtype=tf.float64, shape=[batch_size, 3], name='head_vec')
    tail_vec = tf.placeholder(dtype=tf.float64, shape=[batch_size, 3], name='tail_vec')
    y_true_vec = tf.placeholder(dtype=tf.float64, shape=[batch_size,1], name='y_true')
    #rel_vec = tf.placeholder(dtype=tf.float32, shape=[None,3],name='relations')

    #relation = list(itertools.repeat([1, 0, 0], batch_size))  # Neutral relation
    #relation = [1,1,1]
    #relation = np.random.rand(batch_size,50)
    #print(relation.shape)
    #relation = tf.Variable(tf.random_normal(shape=[None, 50]), name='relation')
    #Failed to convert object of type <class 'list'> to Tensor. Contents: [None, 50]. Consider casting elements to a supported type
    y_pred,dot = DNN(head_vec, tail_vec)
    #loss = loss(y_true=labels_vec, y_pred=y_pred)
    cost,train_op,lear = train(y_true=y_true_vec, y_pred=y_pred, starter_learning_rate=lr)
    print_loss=[]
    learning = []
    epochs = 50
    print('..............Starting session')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            print('epoch = ', epoch)
            for i in range(0, 20, batch_size):
                head = data_vec.iloc[i:i+batch_size, 1:4]
                head = np.asarray(head)
                #head = np.expand_dims(head, axis=1)
                tail = data_vec.iloc[i:i+batch_size, 4:7]
                tail = np.asarray(tail)
                #tail = np.expand_dims(tail, axis=1)
                y_true = data_vec.iloc[i:i+batch_size,7]
                y_true = np.expand_dims(y_true,axis=1)
                #print(head)
                #print(tail)
                #print(y_true)
                #relation_vec = [items for items in itertools.repeat([0,0,0], times=batch_size)]
                #print(relation_vec)
                l,t,prediction,d,le = sess.run([cost,train_op,y_pred,dot,lear],feed_dict={head_vec:head, tail_vec:tail,
                                                                        y_true_vec:y_true})

                epoch_loss += l
                learning.append(le)
                print('Batch = ',i,' to ', i+batch_size ,' loss = ',l)
                #print('pred = ',prediction)
                print('gradients = ',t)
            print_loss.append(epoch_loss)
            print('epoch_loss = ',epoch_loss)


        print('final weights = ',d)
        for i in range(0,10,batch_size):
            head = data_vec.iloc[i:i+batch_size, 1:4]
            head = np.asarray(head)
            #head = np.expand_dims(head, axis=1)
            tail = data_vec.iloc[i:i+batch_size, 4:7]
            tail = np.asarray(tail)
            #tail = np.expand_dims(tail, axis=1)
            labels = data_vec.iloc[i:i+batch_size, 7]
            prediction = sess.run([y_pred], feed_dict={head_vec: head, tail_vec: tail})
            print('testing = ', prediction)


import matplotlib.pyplot as plt
plt.plot(print_loss)
plt.show()



















