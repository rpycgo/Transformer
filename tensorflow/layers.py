import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class PositionalEncoding(Layer):
    '''
    d_model: embedding dimension
    i: index
    n: sequence length
    '''
    def __init__(self, n, d_model):
        super(PositionalEncoding, self).__init__()
        self.n = n
        self.d_model = d_model
        self.positional_encoding = self.positional_encoding()

    def _angles(self, positions, i):
        _angle = 1 / tf.pow(10000, (2 * (i//2) / tf.cast(self.d_model, dtype=tf.float32)))
        pe_i = positions * _angle

        return pe_i

    def positional_encoding(self):
        sequences = tf.range(self.n, dtype=tf.float32)
        positions = tf.expand_dims(sequences, axis=1)
        i = tf.range(self.d_model, dtype=tf.float32)
        i = tf.expand_dims(i, axis=0)
        angles = self._angles(
            positions = positions,
            i = i
        )

        sines = tf.math.sin(angles[:, 0::2])
        cosines = tf.math.cos(angles[:, 1::2])
        angles = np.zeros(angles.shape)
        angles[:, 0::2] = sines
        angles[:, 1::2] = cosines
        angles = tf.Variable(angles, dtype=tf.float32)
        angles = tf.expand_dims(angles, axis=0)

        return angles

    def call(self, x):
        return x + self.positional_encoding[:, :tf.shape(x)[1], :]


class ScaledDotProductAttention(Layer):
    '''
    query: batch_size, num_haed, sequence_length, d_model/num_heads
    key: batch_size, num_haed, sequence_length, d_model/num_heads
    value: batch_size, num_haed, sequence_length, d_model/num_heads
    padding_mask: batch_size, 1, 1, sequence_length
    '''
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def call(self, query, key, value, masking=None):
        matmul_qk = tf.matmul(query, key, transpose_b=True) # batch_size, num_head, sequence_length, sequence_length

        d_k = tf.cast(tf.shape(query)[-1], dtype=tf.float32)
        attention_weights = matmul_qk / tf.math.sqrt(d_k)

        if masking is not None:
            attention_weights += (mask * -1e10)

        logits = tf.nn.softmax(attention_weights, axis=-1)  # batch_size, num_heads, query_length, key_length
        attention_weights = tf.matmul(logits, value) # batch_size, num_heads, query_length, d_model/num_heads

        return attention_weights, logits


