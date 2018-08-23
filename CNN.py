import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from sklearn.metrics import f1_score

# Set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
n_filters = 250
kernel_size = 3
hidden_dims = 250
learning_rate = 0.003
training_steps = 2
width = 3

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# Initialize placeholders
X = tf.placeholder("int32", [None, maxlen])
Y = tf.placeholder("float32", [None, ])

# Xavier initializer for tf.Variables
xavier_init = tf.contrib.layers.xavier_initializer()
word_embs = tf.Variable(xavier_init([max_features, embedding_dims]))
filters = tf.Variable(xavier_init([width, embedding_dims, n_filters]))


def neural_net(x):
    x = tf.nn.embedding_lookup(word_embs, x)
    x = tf.layers.dropout(inputs=x, rate=0.2)
    x = tf.nn.conv1d(value=x, filters=filters, stride=1, padding='VALID')
    x = tf.nn.relu(features=x)
    x = tf.layers.max_pooling1d(inputs=x, pool_size=maxlen - 2, strides=1,
                                padding='VALID')
    x = tf.squeeze(x, [1])
    x = tf.layers.dense(inputs=x, units=250, activation='relu')
    x = tf.layers.dropout(inputs=x, rate=0.2)
    x = tf.layers.dense(inputs=x, units=1)
    return x


logits = tf.squeeze(neural_net(X), [1])
batch_prediction = tf.nn.sigmoid(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables and run interactive session:
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

for step in range(1, training_steps + 1):
    i = 0
    while i * batch_size < len(x_train):
        x_batch, y_batch = x_train[i * batch_size:(i + 1) * batch_size], y_train[i * batch_size:(i + 1) * batch_size]
        i += 1
        # print(i, x_batch.shape, y_batch.shape)

        sess.run(train_op, feed_dict={X: x_batch, Y: y_batch})


print("Optimization Finished!")

# Collect all batch predictions on test dataset:
prediction = np.array([])
i = 0
while i * batch_size < len(x_test):
    x_batch = x_test[i * batch_size:(i + 1) * batch_size]
    y_batch = y_test[i * batch_size:(i + 1) * batch_size]
    i += 1
    a = sess.run(batch_prediction, feed_dict={X: x_batch, Y: y_batch})
    prediction = np.append(prediction, np.asarray(a))

# Obtain label predictions by rounding predictions to int
prediction = [int(round(t)) for t in prediction]

# Use F1 metric:
F1 = f1_score(y_true=y_test, y_pred=prediction, average=None)
print("F1 score: ", F1)

sess.close()
