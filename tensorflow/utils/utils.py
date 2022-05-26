import tensorflow as tf


def is_padded(x: tf.Tensor) -> tf.Tensor:
    masking = tf.cast(tf.math.equal(x, 0), tf.float32)   # batch_size, 1, 1, seq_len
    masking = masking[:, tf.newaxis, tf.newaxis, :]
    
    return masking
