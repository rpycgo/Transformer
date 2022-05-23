import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Permute


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


class MultiHeadAttention(Layer):
    '''
    d_model: embedding dimension
    h: head numbers
    '''
    def __init__(self, d_model, h):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        
        self.d_k = d_model // h

        self.linear_query = Dense(units=d_model)
        self.linear_key = Dense(units=d_model)
        self.linear_value = Dense(units=d_model)
        self.linear_output = Dense(units=d_model)

    def _split_by_heads(self, x):
        x = tf.reshape(x, shape=(-1, x.shape[1], self.h, self.d_k))
        x = Permute((2, 1, 3))(x)

        return x

    def call(self, query, key, value, mask=None):
        query_projected = self.linear_query(query)  # batch_size, query_len, d_model
        key_projected = self.linear_key(key)        # batch_size, key_len, d_model
        value_projected = self.linear_value(value)  # batch_size, value_len, d_model

        query_splitted = self._split_by_heads(query)   # batch_size, h, query_len, d_model/h
        key_splitted = self._split_by_heads(key)       # batch_size, h, key_len, d_model/h
        value_splitted = self._split_by_heads(value)   # batch_size, h, value_len, d_model/h

        scaled_dot_product_attention, _ = ScaledDotProductAttention()(query_splitted, key_splitted, value_splitted, mask)  # batch_size, h, query_len, d_model/h
        scaled_dot_product_attention = Permute((2, 1, 3))(scaled_dot_product_attention) # batch_size, query_len, h, d_model/h
        
        # concat layer
        concat_scaled_dot_product_attention = tf.reshape(scaled_dot_product_attention, shape=(-1, query.shape[1], self.d_model))  # batch_size, query_len, d_model

        outputs = self.linear_output(concat_scaled_dot_product_attention)

        return outputs
