import tensorflow as tf

class CNN():
    def __init__(self, n_embedding, n_vocab, sent_max_features, sent_maxlen, n_filters,
                 sent_kernel_size, sent_hidden_dims, sent_learning_rate, sent_width, sent_eval_batch_size): #attention_size is added
        #ATTENTION.__init__(self,embedding, n_embedding, n_vocab, attention_size)
        # params
        self.n_embedding = n_embedding
        self.n_vocab = n_vocab
        self.own_embedding = tf.Variable(tf.random_uniform((self.n_vocab, self.n_embedding), -1, 1, seed=123))


        self.max_features = sent_max_features
        self.maxlen = sent_maxlen
        self.n_filters = n_filters
        self.kernel_size = sent_kernel_size
        self.hidden_dims = sent_hidden_dims
        self.learning_rate = sent_learning_rate
        self.width = sent_width
        self.eval_batch_size = sent_eval_batch_size
        # graph nodes

        self.X = tf.placeholder("int32", [None, self.maxlen])
        self.Y = tf.placeholder("float32", [None, ])

        #self.attention = False
        #self.lookup_matrix = embedding #if not self.attention else self.attented_embs

        self.xavier_init = tf.contrib.layers.xavier_initializer(seed = 741)
        #                    word_embs = tf.Variable(xavier_init([max_features, embedding_dims]))
        self.filters = tf.Variable(self.xavier_init([self.width, self.n_embedding, self.n_filters]))  # embedding_dims
        self.logits = tf.squeeze(self.neural_net(), [1])
        self.batch_prediction = tf.nn.sigmoid(self.logits)


        # Define loss and optimizer
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y)) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.train = self.optimizer.minimize(self.loss, var_list=(self.filters, self.own_embedding))


    def neural_net(self):
        x = tf.nn.embedding_lookup(self.own_embedding, self.X)
        x = tf.layers.dropout(inputs=x, rate=0.2)
        x = tf.nn.conv1d(value=x, filters=self.filters, stride=1, padding='VALID')
        x = tf.nn.relu(features=x)
        x = tf.layers.max_pooling1d(inputs=x, pool_size=self.maxlen - 2, strides=1,
                                    padding='VALID')
        x = tf.squeeze(x, [1])
        x = tf.layers.dense(inputs=x, units=self.hidden_dims, activation='relu')
        x = tf.layers.dropout(inputs=x, rate=0.2)
        x = tf.layers.dense(inputs=x, units=1)
        return x

