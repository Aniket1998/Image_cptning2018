# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
print(tf.VERSION)


# In[10]:


xavier = tf.contrib.layers.xavier_initializer()
zeros = tf.zeros_initializer()
dim_hidden = 1024 #Dim of hidden layer
dim_vocabulary = 25000
vocabulary_size = 25000 #TODO : FIT VOCABULARY SIZE
dim_embed = 512
embedding_size = 512 #Size of the word embeddings
annotation_L = 196 #Annotation vectors are of the form L*D
annotation_D = 512
max_timesteps = 16
lamda = 1.0
dropout1 = True
dropout1_rate = 0.5
dropout2 = True
dropout2_rate = 0.5
consider_z = True
consider_y = True
batch_size=196
alpha_c = 1
NULL = 1
start = 0


# In[5]:


def initial_c0_h0(annotations):
    with tf.variable_scope('initial_c0_h0'):
        feature_mean = tf.reduce_mean(annotations,axis=1)
        c0 = tf.layers.dense(inputs=feature_mean,units=dim_hidden,activation=tf.nn.tanh,kernel_initializer=xavier)
        h0 = tf.layers.dense(inputs=feature_mean,units=dim_hidden,activation=tf.nn.tanh,kernel_initializer=xavier)
        return c0,h0 #N * dim_hidden both


# In[9]:


#UNDER CONSTRUCTION
def soft_attention(annotations,h_prev,i):
    with tf.variable_scope('soft_attention',reuse=tf.AUTO_REUSE):
        #HANDLE WITH CARE
        h_extend = tf.expand_dims(h_prev,axis=1) # N * 1 * h
        onemat = tf.ones(shape=[batch_size,annotation_L,1]) # N * L * 1
        h_large = tf.matmul(onemat,h_extend) # N * L * h
        joined = tf.reshape(tf.concat([annotations,h_large],axis=2),[-1,dim_hidden + annotation_D]) # N  L * D + h
        e = tf.reshape(tf.layers.dense(inputs=joined,units=1,activation=tf.nn.relu,kernel_initializer=xavier),[-1,annotation_L])
        #HANDLE WITH CARE ENDS
        alpha = tf.nn.softmax(e) #dimension N * L
        # print(alpha)
        beta = tf.layers.dense(inputs=h_prev,units=1,activation=tf.nn.sigmoid) #N * 1
        alpha_exp = tf.expand_dims(alpha,axis=1) #N * 1 * L
        pre_gating = tf.reshape(tf.matmul(alpha_exp,annotations),[-1,annotation_D]) # (N,1,L) * (N,L,D) = (N,1,D) reshaped to (N,D)
        z = tf.multiply(beta,pre_gating) #N * 1 and N * D pointwise multiplication 
        return z,alpha,beta


# In[4]:


def get_annotation(img_features):
    with tf.variable_scope('get_annotation_vecs'):
        W = tf.get_variable('W',[annotation_D,annotation_D])
        features_flat = tf.reshape(img_features,[-1,annotation_D])
        features_proj = tf.matmul(features_flat,W)
        annotations = tf.reshape(features_proj,[-1,annotation_L,annotation_D])
        return annotations # N * L * D


# In[8]:


def word_embedding(one_hot,i):
    with tf.variable_scope('word_embedding',reuse=tf.AUTO_REUSE):
        word_embedding = tf.get_variable('word_embedding',[vocabulary_size,embedding_size],initializer=tf.random_uniform_initializer(minval=-1.0,maxval=1.0))
        return tf.nn.embedding_lookup(word_embedding,one_hot)


# In[11]:


def get_logits(ey,h,z,training,i):
    with tf.variable_scope('get_logits',reuse=tf.AUTO_REUSE):
        logits = h
        # print(logits)
        if dropout1:
            logits = tf.layers.dropout(inputs=logits,rate=dropout1_rate,training=training)
        # print(logits)
        logits = tf.layers.dense(inputs=logits,units=dim_embed,activation=None,kernel_initializer=xavier)
        # print(logits)
        if consider_y:
            logits += ey
        # print(logits)
        if consider_z:
            logits += tf.layers.dense(inputs=logits,units=dim_embed,activation=None,use_bias=False,kernel_initializer=xavier)
        # print(logits)
        logits = tf.nn.tanh(logits)
        # print(logits)
        if dropout2:
            logits = tf.layers.dropout(inputs=logits,rate=dropout2_rate,training=training)
        # print(logits)
        return tf.layers.dense(inputs=logits,units=dim_vocabulary,activation=None,kernel_initializer=xavier)


# In[3]:


def model_function(features,labels,mode,params):
    # print(features)
    # print(labels)
    img_features = tf.feature_column.input_layer(features, params['feature_columns'])
    # print(img_features)
    a = get_annotation(img_features)
    c,h = initial_c0_h0(a)
    cp = c
    hp = h#for prediction
    cell = rnn.BasicLSTMCell(num_units=dim_hidden)
    start = tf.zeros([batch_size],dtype=tf.int32)
    lastword = word_embedding(start,1)
    alpha_list = []
    predictions = []
    loss = 0.0
    for t in range(max_timesteps):
        zp,alphap,betap = soft_attention(a,hp,1)
        # print(lastword)
        print(zp)
        _,[cp,hp] = cell(tf.concat([lastword, zp],1),[cp, hp])
        logitsp = get_logits(lastword,hp,zp,False,1)
        prediction = tf.argmax(logitsp,axis=1)
        predictions.append(prediction)
        lastword = word_embedding(prediction,1)
    predictions = tf.stack(predictions)
    predictions = tf.transpose(predictions)
    print(predictions)
    predictions = {
        'sentences' : predictions
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(predictions=predictions,mode=mode)
    mask = tf.to_float(tf.not_equal(labels, NULL))
    x = word_embedding(labels,1)
    for t in range(max_timesteps):
        z,alpha,beta = soft_attention(a,h,t)
        alpha_list.append(alpha)
        # print(x)
        _,[c,h] = cell(tf.concat( [x[:,t,:], z],1),[c, h])
        # print(h)
        logits=get_logits(x[:,t,:],h,z,True,t)
        # print(logits)
        loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[:, t+1],logits=logits)*mask[:, t+1] )
    alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))
    alphas_all = tf.reduce_sum(alphas, 1)
    alpha_reg = alpha_c * tf.reduce_sum((max_timesteps/batch_size - alphas_all) ** 2)
    loss += alpha_reg
    loss = loss / tf.to_float(batch_size)
    # tf.Print(loss,[loss],message="Loss is : ")
    eval_ops = {
        'accuracy' : tf.metrics.accuracy(labels=labels[:,1:-1],predictions=predictions['sentences'])
    }
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_ops)
