import keras.backend as K
import copy
import tensorflow as tf

def loss_mse():
    def n2v_mse(y_true, y_pred):
        target, mask = tf.split(y_true, 2, axis=len(y_true.shape)-1)
        loss = tf.reduce_sum(K.square(target - mask*y_pred)) / tf.reduce_sum(mask)
        return loss

    return n2v_mse

def loss_mae():
    def n2v_abs(y_true, y_pred):
        target, mask = tf.split(y_true, 2, axis=len(y_true.shape)-1)
        loss = tf.reduce_sum(K.abs(target - y_pred*mask)) / tf.reduce_sum(mask)
        return loss

    return n2v_abs

# Make Gaussian kernel following SciPy logic
def make_gaussian_2d_kernel(sigma, truncate=4.0, dtype=tf.float32):
    radius = round(sigma * truncate)
    x = tf.cast(tf.range(-radius, radius + 1), dtype=dtype)
    k = tf.exp(-0.5 * tf.square(x / sigma))
    k = k / tf.reduce_sum(k)
    return tf.expand_dims(k, 1) * k

def loss_DIP():
    def n2v_DIP(y_true, y_pred):
        e=0.2
        target, mask = tf.split(y_true, 2, axis=len(y_true.shape)-1)
        # Convolution kernel
        kernel = make_gaussian_2d_kernel(8)
        # Apply kernel to each channel
        kernel = tf.tile(kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, 1, 1])
        processed_target = tf.nn.separable_conv2d(
            target, kernel, tf.eye(1, batch_shape=[1, 1]),
            strides=[1, 1, 1, 1], padding='SAME')
        final_y_pred = (1-e)*y_pred*mask + e*processed_target
        loss = tf.reduce_sum(K.square(target - final_y_pred)) / tf.reduce_sum(mask)
        return loss

    return n2v_DIP