import numpy as np
import tensorflow as tf

#VEL_PRIORITY = 50
VEL_PRIORITY = 250

# prioritize velocity, via magnification
loss_scale = tf.convert_to_tensor(
    np.array(
        [[
            [1] * 3, 
            [VEL_PRIORITY] * 3
        ]], 
        dtype=np.float32
    )
)

def custom_loss(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(
        y_true * loss_scale, 
        y_pred * loss_scale
    )

# data = np.load("data/ANN_06_temp.npy")
# pos_std = np.std(data[..., 0, :])
# vel_std = np.std(data[..., 1, :])
pos_std = 4068.2464010399176
vel_std = 4.426527283948402

scale_mat_line = np.array(
    [[pos_std] * 3, [vel_std] * 3], 
    dtype=np.float32
)


norm = tf.keras.layers.Dense(6)
norm(tf.convert_to_tensor(np.zeros((10, 6), dtype=np.float32)))
norm.set_weights([tf.convert_to_tensor(np.diagflat(1 / scale_mat_line)), norm.get_weights()[1]])
norm.trainable = False

inv_norm = tf.keras.layers.Dense(6)
inv_norm(tf.convert_to_tensor(np.zeros((10, 6), dtype=np.float32)))
inv_norm.set_weights([tf.convert_to_tensor(np.diagflat(scale_mat_line)), inv_norm.get_weights()[1]])
inv_norm.trainable = False


def accuracy(y_true, y_pred):
    return np.mean(np.sqrt(np.sum((y_true - np.array(y_pred, dtype=np.float32)) ** 2, axis=2)), axis=0)

def load_model(filename):
    return tf.keras.models.load_model(filename, custom_objects={"custom_loss":custom_loss})