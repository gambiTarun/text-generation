#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time


# In[6]:


data = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


# In[7]:


text = open(data).read()
# lets look at the text
print(text[:1000])


# In[8]:


# set-> Build an unordered collection of unique elements.
vocab = sorted(set(text))
print(vocab)
print('Number of unique chars: {}'.format(len(vocab)))


# In[9]:


# creating a mapping from unique chars to index and vice versa
idx_from_char = {u:i for i,u in enumerate(vocab)}
char_from_idx = np.array(vocab)

text_as_int = np.array([idx_from_char[c] for c in text])


# In[11]:


print('{} ------mapped--to-----> {}'.format(text[:13], text_as_int[:13]))


# In[10]:


# Creating training samples
# the maximum length sentence we want for a single input in characters
seq_length = 100

# breaking the text into chunks of seq_lenght+1
chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_length+1, drop_remainder=True)

# text broken into chunks of seq_lenght+1 parts to create training and target data
for item in chunks.take(5):
    print(repr(''.join(char_from_idx[item.numpy()])))
    
# example: 
# chunk = 'Apple' (seq_lenght+1 = 5)
# training point = 'Appl'
# target point = 'pple'


# In[12]:


def split_train_target(chunk):
    train = chunk[:-1]
    target = chunk[1:]
    return train, target

dataset = chunks.map(split_train_target)


# In[13]:


for in_example, out_example in dataset.take(1):
    print('input string: ', repr(''.join(char_from_idx[in_example.numpy()])))
    print('target string: ', repr(''.join(char_from_idx[out_example.numpy()])))


# In[14]:


for i,(in_idx, out_idx) in enumerate(zip(in_example[:5],out_example[:5])):
    print('Step {:2d}'.format(i))
    print('    input: {} ({:s})'.format(in_idx,char_from_idx[in_idx]))
    print('    expected out: {} ({:s})'.format(out_idx,char_from_idx[out_idx]))


# In[15]:


# Batch size 
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences, 
# so it doesn't attempt to shuffle the entire sequence in memory. Instead, 
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset_shuffled = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# text broken into chunks of seq_lenght+1 parts to create training and target data
# then dataset of 'in' and 'out' created
# now SHUFFLED
for in_example, out_example in dataset_shuffled.take(1):
    print('input dataset shape: ', in_example.shape)
    print('target dataset shape: ', out_example.shape)
    # This Shape will be the inpput and output shape of our Model


# In[17]:


# The MODEL (using Functional API)
class Model(tf.keras.Model):
    def __init__(self,vocab_size, embedding_dim, units):
        super(Model, self).__init__()
        self.units = units
        self.embedding  = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       recurrent_activation='sigmoid',
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform',
                                       stateful=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
    def call(self, x):
        embedding = self.embedding(x)
        output = self.gru(embedding)
        prediction = self.fc(output)
        return prediction

vocab_size = len(vocab)   # (y=[vocab_size,1], Wya=[vocab_size,units])
embedding_dim = 256       # (x=[embedding_dim,1], Wax=[embedding_dim,units])
units = 1024              # units are the number of nodes in the kernel layer (a=[units,1] ,Waa=[units,units])

model = Model(vocab_size,embedding_dim,units)

# OR
#
# This approach requires you to enter the specifications of the 
# model (if there, like number of units in wach layer) with the 
# inputs when creating a model 
# 
# The above method can solve this problem by using a constructor to use the 
# specifications when creating an instance of the model, which later can be
# called with inputs to create the model. 
# 
# def Model_2(inputs,vocab_size,embedding_dim,units):
#     embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
#     x = tf.keras.layers.GRU(units,
#                                  recurrent_activation='sigmoid',
#                                  return_sequences=True,
#                                  recurrent_initializer='glorot_uniform',
#                                  stateful=True)(embedding)
#     outputs = tf.keras.layers.Dense(vocab_size)(x)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

# model = Model_2(inputs,vocab_size,embedding_dim,units)


# In[18]:


optimizer = tf.train.AdamOptimizer()

# using sparse_.. so that we do not nedd to convert to one-hot vectors
def loss_func(real,pred):
    return  tf.losses.sparse_softmax_cross_entropy(labels=real, logits=pred)


# In[19]:


# The above method of using a SubClass to instantiate the Model
# require us to 'build' to show the shape of the input layer

model.build(tf.TensorShape([BATCH_SIZE, seq_length]))


# In[26]:


model.summary()


# In[21]:


model.variables


# In[22]:


# Directory where the checkpoints will be saved
checkpoint_dir = 'C:/Users/Tarunbir Singh/Documents/Machine Learning/text_generation/training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
# Checkpoint instance
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


# In[88]:


epochs = 30

for epoch in range(epochs):
    start = time.time()
    
    # initialising the hidden state at the start of every epoch
    # initial hidden state is None
    hidden = model.reset_states()
    
    for (batch, (inp,target)) in enumerate(dataset_shuffled):
        with tf.GradientTape() as tape:
            # feeding the hidden state back into the model
            predictions = model(inp)
            loss = loss_func(target, predictions)
            
        # gradients for back-propagation using Gradient Taping
        grads = tape.gradient(loss, model.variables)
        # applying the updation of variables using the calcuated grads
        optimizer.apply_gradients(zip(grads,model.variables))
        
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,batch,loss))
            
    # saving (checkpoint) the model every 5 epochs
    if (epoch+1)%5==0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# In[23]:


# model needs to train only once
# can be restored from the checkpoints created
model = Model(vocab_size, embedding_dim, units)

checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1,None]))


# In[25]:


# Evaluation Step

# number of characters to generate
num_generate = 1000

# The Starting String input, 
# can be experimented
starting_string = 'f'

# convert to index our starting string
input_eval = [idx_from_char[i] for i in starting_string]
input_eval = tf.expand_dims(input_eval, 0)

# variable to store the generated text
text_generated = []

# Low temperatures results in more predictable text.
# Higher temperatures results in more surprising text.
# Experiment to find the best setting.
temperature = 1.0

# here batch size == 1
model.reset_states()
for i in range(num_generate):
    prediction = model(input_eval)
    # remove the batch dimensions (flatten)
    prediction = tf.squeeze(prediction, 0)
    
    # using a multinomial distribution to predict the word returned by the model
    prediction = prediction / temperature
    predicted_id = tf.multinomial(prediction, num_samples=1)[-1,0].numpy()
    
    # we pass the predicted word as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id],0)
    
    text_generated.append(char_from_idx[predicted_id])
    
print(starting_string + ''.join(text_generated))
    


# In[ ]:




