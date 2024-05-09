import numpy as np
import tensorflow as tf

# prioritize velocity, via magnification
loss_scale = tf.convert_to_tensor(
    np.array(
        [[
            [1], 
            [100]
        ]], 
        dtype=np.float32
    )
)


def custom_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(
        y_true * loss_scale, 
        y_pred * loss_scale
    )