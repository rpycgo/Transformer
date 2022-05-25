import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Permute, Dropout, LayerNormalization, Embedding, Lambda
from tensorflow.keras.models import Model


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
            attention_weights += (masking * -1e10)

        logits = tf.nn.softmax(attention_weights, axis=-1)  # batch_size, num_heads, query_length, key_length
        attention_weights = tf.matmul(logits, value) # batch_size, num_heads, query_length, d_model/num_heads

        return attention_weights, logits


class MultiHeadAttention(Layer):
    '''
    d_model: embedding dimension
    h: head numbers
    '''
    def __init__(self, d_model, h, max_len, name='multi-head_attention'):
        super(MultiHeadAttention, self).__init__(name)
        self.d_model = d_model
        self.h = h
        self.max_len = max_len
        
        self.d_k = d_model // h

        self.linear_query = Dense(units=d_model)
        self.linear_key = Dense(units=d_model)
        self.linear_value = Dense(units=d_model)
        self.linear_output = Dense(units=d_model)

    def _split_by_heads(self, x):
        x = tf.reshape(x, shape=(-1, self.max_len, self.h, self.d_k))
        x = Permute((2, 1, 3))(x)

        return x

    def call(self, query, key, value, masking=None):
        query_projected = self.linear_query(query)  # batch_size, query_len, d_model
        key_projected = self.linear_key(key)        # batch_size, key_len, d_model
        value_projected = self.linear_value(value)  # batch_size, value_len, d_model

        query_splitted = self._split_by_heads(query)   # batch_size, h, query_len, d_model/h
        key_splitted = self._split_by_heads(key)       # batch_size, h, key_len, d_model/h
        value_splitted = self._split_by_heads(value)   # batch_size, h, value_len, d_model/h

        scaled_dot_product_attention, _ = ScaledDotProductAttention()(query_splitted, key_splitted, value_splitted, masking)  # batch_size, h, query_len, d_model/h
        scaled_dot_product_attention = Permute((2, 1, 3))(scaled_dot_product_attention) # batch_size, query_len, h, d_model/h
        
        # concat layer
        concat_scaled_dot_product_attention = tf.reshape(scaled_dot_product_attention, shape=(-1, self.max_len, self.d_model))  # batch_size, query_len, d_model

        outputs = self.linear_output(concat_scaled_dot_product_attention)

        return outputs


class EncoderNetwork(Model):
    def __init__(self, max_len, d_model=512, h=8, d_ff=2048, dropout_rate=0.1, name='encoder'):
        super(EncoderNetwork, self).__init__(name)
        self.max_len = max_len
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        self.multi_head_attention = MultiHeadAttention(self.d_model, self.h, self.max_len)
        self.feed_forward_output = Dense(units=self.d_ff, activation='relu')
        self.dimension_adjusting = Dense(units=self.d_model, use_bias=False)

    def call(self, input, masking):
        multi_head_attention = self.multi_head_attention(
            query=input,
            key=input,
            value=input,
            masking=masking
            )
        multi_head_attention = Dropout(rate=self.dropout_rate)(multi_head_attention)
        multi_head_attention = LayerNormalization()(input + multi_head_attention)

        feed_forward = self.feed_forward_output(multi_head_attention)
        feed_forward = self.dimension_adjusting(feed_forward)
        feed_forward = Dropout(rate=self.dropout_rate)(feed_forward)

        outputs = LayerNormalization()(multi_head_attention + feed_forward)

        return outputs


class Encoder:
    def __init__(self, vocab_size, max_len, num_encoders=6, d_model=512, h=8, d_ff=2048, dropout_rate=0.1, name='encoder'):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_encoders = num_encoders
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
    def __call__(self):
        input = Input(shape=(None,))
        masking = Input(shape=(1, 1, None))

        input_embedding = Embedding(self.vocab_size, self.d_model)(input)
        input_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # weights * sqrt d_model
        input_embedding = PositionalEncoding(self.vocab_size, self.d_model)(input_embedding)
        output = Dropout(rate=self.dropout_rate)(input_embedding)

        for i in range(self.num_encoders):
            output = EncoderNetwork(
                max_len=self.max_len,
                d_model=self.d_model,
                h=self.h,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                name=f'encoder_{i+1}'
                )(output, masking)

        model = Model(
            inputs=[input, masking],
            outputs=output,
            name='encoder'
        )

        return model


class DecoderNetwork(Model):
    def __init__(self, max_len, d_model=512, h=8, d_ff=2048, dropout_rate=0.1, name='decoder'):
        super(DecoderNetwork, self).__init__(name)
        self.max_len = max_len
        self.d_model = d_model
        self.h = h
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        self.multi_head_attention = MultiHeadAttention(self.d_model, self.h, self.max_len)
        self.feed_forward_output = Dense(units=self.d_ff, activation='relu')
        self.dimension_adjusting = Dense(units=self.d_model, use_bias=False)

    def call(self, input, encoder_output, masking_for_leftward, masking):
        masked_multi_head_attention = self.multi_head_attention(
            query=input,
            key=input,
            value=input,
            masking=masking_for_leftward,
            # name='masked_multi-head_attention'
            )
        masked_multi_head_attention = Dropout(rate=self.dropout_rate)(masked_multi_head_attention)
        masked_multi_head_attention = LayerNormalization()(input + masked_multi_head_attention)

        multi_head_attention = self.multi_head_attention(
            query=masked_multi_head_attention,
            key=encoder_output,
            value=encoder_output,
            masking=masking,
            # name='multi-head_attention'
            )
        multi_head_attention = Dropout(rate=self.dropout_rate)(multi_head_attention)
        multi_head_attention = LayerNormalization()(masked_multi_head_attention + multi_head_attention)       

        feed_forward = self.feed_forward_output(multi_head_attention)
        feed_forward = self.dimension_adjusting(multi_head_attention)
        feed_forward = Dropout(rate=self.dropout_rate)(feed_forward)

        outputs = LayerNormalization()(multi_head_attention + feed_forward)

        return outputs


