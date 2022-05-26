import tensorflow as tf


def is_padded(x: tf.Tensor) -> tf.Tensor:
    masking = tf.cast(tf.math.equal(x, 0), tf.float32)   # batch_size, 1, 1, seq_len
    masking = masking[:, tf.newaxis, tf.newaxis, :]
    
    return masking


def mask_future_data(x: tf.Tensor) -> tf.Tensor:
    seq_len = x.shape[1]
    square_matrix = tf.ones((seq_len, seq_len))
    future_masking_matrix = 1 - tf.linalg.band_part(square_matrix, -1, 0)   # upper triangle except diagonal
    padding_check_matrix = is_padded(x)

    return tf.maximum(future_masking_matrix, padding_check_matrix)
