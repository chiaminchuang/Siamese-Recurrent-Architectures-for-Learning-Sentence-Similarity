import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
class SiameseLSTM():
    def __init__(self, config, sess, embeddings is_training=True):
        
        num_step = config['max_len']
        emb_dim = config['emb_dim']
		vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        
        self.sentence_A = tf.placeholder(tf.int32, [None, num_step])
        self.sentence_B = tf.placeholder(tf.int32, [None, num_step])
        self.mask_A = tf.placeholder(tf.float64, [None, num_step])
        self.mask_B = tf.placeholder(tf.float64, [None, num_step])
        self.relatedness = tf.placeholder(tf.float64, [None])
        
        self.batch_size = tf.Variable(0, dtype=tf.int32, trainable=False)
        
        with tf.name_scope('Embedding_Layer'):
            embedding_initializer = tf.constant_initializer(embeddings, dtype=tf.float64)
            embedding_weights = tf.get_variable(dtype=tf.float64, name='embedding_weights',shape=(vocab_size, emb_dim), initializer=embedding_initializer, trainable=False)
            self.embedded_A = tf.nn.embedding_lookup(embedding_weights, self.sentence_A) # (batch_size, num_step, emb_dim)
			self.embedded_B = tf.nn.embedding_lookup(embedding_weights, self.sentence_B) # (batch_size, num_step, emb_dim)
            
            
        with tf.name_scope('LSTM_Output'):
            self.outputs_A = self.LSTM(sequence=self.embedded_A, reuse=None) # (batch_size, num_step, emb_dim)
            self.outputs_B = self.LSTM(sequence=self.embedded_B, reuse=True) # (batch_size, num_step, emb_dim)
            self.masked_outputs_A = tf.reduce_sum(self.outputs_A * self.mask_A[:, :, None], axis=1) # (batch_size, emb_dim)
            self.masked_outputs_B = tf.reduce_sum(self.outputs_B * self.mask_B[:, :, None], axis=1) # (batch_size, emb_dim)
            
        with tf.name_scope('Similarity'):
            self.diff = tf.reduce_sum(tf.abs(tf.subtract(self.masked_outputs_A, self.masked_outputs_B)), axis=1) # 32
            self.similarity = tf.clip_by_value(tf.exp(-1.0 * self.diff), 1e-7, 1.0 - 1e-7)
        
        # MSE
        with tf.name_scope('Loss'):
            diff = tf.subtract(self.similarity, tf.clip_by_value((self.relatedness - 1.0) / 4.0, 1e-7, 1.0 - 1e-7)) # 32
            self.loss = tf.square(diff) # (batch_size,)
            self.cost = tf.reduce_mean(self.loss) # (1,)
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.Variable(0.0, trainable=False)
        
        train_variables = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, train_variables), config['max_grad_norm'])
        
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1, epsilon=1e-6)
        
        with tf.name_scope('Train'):
#             self.train_op = optimizer.minimize(self.cost)
            self.train_op = optimizer.apply_gradients(zip(gradients, train_variables))
        
        
            
    def LSTM(self, sequence, reuse=None):
        def sequence_length(sequence):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=2))
            length = tf.reduce_sum(used, axis=1)
            return tf.cast(length, tf.int32)
            
        with tf.variable_scope('LSTM', reuse=reuse, dtype=tf.float64):
            cell = LSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
        
        seq_len = sequence_length(sequence)
        with tf.name_scope('Siamese'), tf.variable_scope('Siamese', dtype=tf.float64):
            outputs, state = tf.nn.dynamic_rnn(cell, sequence, dtype=tf.float64, sequence_length=seq_len)
        
        return outputs
    
         