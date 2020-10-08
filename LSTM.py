from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import backend as K

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from numpy.random import seed
seed(42)  # keras seed fixing
tf.random.set_seed(42)  # tensorflow seed fixing

import wandb
from wandb.keras import WandbCallback

hyperparameter_defaults = dict(
    scale_data = True,
    window_size = 180,
    weigh_classes = False,
    resample_classes = False
)

wandb.init(project="INFO411-Assignment2", notes="Test", config=hyperparameter_defaults)

# Initial setup
MODEL_NAME = "ECG_LSTM_Large_03"
SEQ_SIZE = wandb.config.window_size * 2 + 1
NUM_CLASSES = 4

df = pd.read_csv("Training_2/outputFull/combined_output_full_baselined_split_{}.csv".format(wandb.config.window_size))

# Drop useless first column, and convert to numpy array
data = df.drop(df.columns[0], axis=1).to_numpy()

X = data[:, :-1]
y = data[:, -1]

X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, test_size=0.3, random_state=42)

if wandb.config.scale_data:
    # Standard scale the entire training set.
    X_flat_train = X_train.flatten().reshape(-1, 1)

    ss = StandardScaler()
    ss.fit(X_flat_train)

    X_train = ss.transform(X_flat_train).reshape(X_train.shape)

    X_flat_val = X_val.flatten().reshape(-1, 1)
    X_val = ss.transform(X_flat_val).reshape(X_val.shape)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

# Potentially add weights to the classes
if wandb.config.weigh_classes:
    class_counts = np.unique(y_train, return_counts=True)[1]
    class_weights = np.full(NUM_CLASSES, class_counts.max() / NUM_CLASSES) / class_counts

    class_weights_dict = dict()
    for idx in range(NUM_CLASSES):
        class_weights_dict.update({idx: class_weights[idx]})
else:
    class_weights_dict = None

# Potentially oversample the minority classes
if wandb.config.resample_classes:
    classes = np.unique(y)

    X_0 = X_train[y_train == classes[0]]
    X_1 = X_train[y_train == classes[1]]
    X_2 = X_train[y_train == classes[2]]
    X_3 = X_train[y_train == classes[3]]
    y_0 = y_train[y_train == classes[0]]
    y_1 = y_train[y_train == classes[1]]
    y_2 = y_train[y_train == classes[2]]
    y_3 = y_train[y_train == classes[3]]

    ids = np.arange(len(X_1))
    choices = np.random.choice(ids, len(X_0))
    sampled_X_1 = X_1[choices]
    sampled_y_1 = y_1[choices]

    ids = np.arange(len(X_2))
    choices = np.random.choice(ids, len(X_0))
    sampled_X_2 = X_2[choices]
    sampled_y_2 = y_2[choices]

    ids = np.arange(len(X_3))
    choices = np.random.choice(ids, len(X_0))
    sampled_X_3 = X_3[choices]
    sampled_y_3 = y_3[choices]

    # Join back together:
    resampled_X_total = np.concatenate([X_0, sampled_X_1, sampled_X_2, sampled_X_3], axis=0)
    resampled_y_total = np.concatenate([y_0, sampled_y_1, sampled_y_2, sampled_y_3], axis=0)

    # Shuffle just to be safe:
    order = np.arange(len(resampled_X_total))
    np.random.shuffle(order)
    X_train = resampled_X_total[order]
    y_train = resampled_y_total[order]


def calc_jk(true_arr, pred_arr):
    k_index = cohen_kappa_score(true_arr, pred_arr)

    cr = classification_report(true_arr, pred_arr, output_dict=True)

    print(cr)

    s_sensitivity = cr['1.0']['recall']
    v_sensitivity = cr['2.0']['recall']

    s_ppv = cr['1.0']['precision']
    v_ppv = cr['2.0']['precision']

    j_index = s_sensitivity + v_sensitivity + s_ppv + v_ppv

    jk_index = (1 / 2) * k_index + (1 / 8) * j_index
    return k_index, j_index, jk_index


def build_model():
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(SEQ_SIZE, 1)))
    model.add(LSTM(100))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(NUM_CLASSES))  # No softmax, because it's done by the loss function
    print(model.summary())
    return model


model = build_model()
# model = tf.keras.models.load_model("Char_model_test.h5", custom_objects={'perplexity': perplexity})
# # latest = tf.train.latest_checkpoint("training_char/")
# # print(latest)
# # model.load_weights(latest)
# # opt = tf.keras.optimizers.Adam()
# # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy', perplexity])
# #
# # model.save("Char_model_test.h5")
#
#
# print(model.summary())
# model.evaluate(gen)
# i1 = gen
#
save_model = tf.keras.callbacks.ModelCheckpoint("saved/" + MODEL_NAME + ".h5", monitor='val_loss', verbose=0,
                                                save_best_only=True, save_weights_only=False, mode='min', period=1)

checkpoint_path = "checkpoints/" + MODEL_NAME + "cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=10)

def log_cm(tag, type, data):
    # print(val_gen.__getitem__(0)[1])
    X_in, y_in = data

    preds = np.argmax(model.predict(X_in), axis=1)

    cm = confusion_matrix(
        y_in,
        preds
    )
    fig, ax = plt.subplots()
    ax.matshow(cm)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, z, ha='center', va='center')

    wandb.log({type + "CM_" + str(tag): fig})
    plt.savefig("CMs/" + type + "CM_" + str(tag) + ".png")

    k_score, j_score, jk_score = calc_jk(y_in, preds)
    wandb.log({type + '_k': k_score, 'epoch': tag})
    wandb.log({type + '_j': j_score, 'epoch': tag})
    wandb.log({type + '_jk': jk_score, 'epoch': tag})


class Give_metrics(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            log_cm(epoch, "val", (X_val, y_val))
            log_cm(epoch, "train", (X_train, y_train))

if True:
    opt = tf.keras.optimizers.Adam()

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # compile model
    model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=32, validation_data=(X_val, y_val), epochs=100, callbacks=[WandbCallback(), save_model, cp_callback, Give_metrics()], shuffle=True, class_weight=class_weights_dict)


# Evaluate on testing data
df = pd.read_csv("Testing_2/outputFull/combined_output_full_baselined_split_{}.csv".format(wandb.config.window_size))

# Drop useless first column, and convert to numpy array
data_test = df.drop(df.columns[0], axis=1).to_numpy()

X_test = data[:, :-1]
y_test = data[:, -1]

if wandb.config.scale_data:
    # Standard scale the entire training set.
    X_flat_test = X_test.flatten().reshape(-1, 1)
    X_test = ss.transform(X_flat_test).reshape(X_test.shape)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Load best model found during training
model = keras.models.load_model("saved/" + MODEL_NAME + ".h5")

# scores = model.evaluate(X_test, y_test)

# for item, name in zip(scores, model.metrics_names):
#     print('Test ' + name, item)
#     wandb.run.summary['Test ' + name] = item


k_score, j_score, jk_score = calc_jk(y_test, np.argmax(model.predict(X_test), axis=1))

print('Test k_score', k_score)
wandb.run.summary['Test k_score'] = k_score

print('Test j_score', j_score)
wandb.run.summary['Test j_score'] = j_score

print('Test jk_score', jk_score)
wandb.run.summary['Test jk_score'] = jk_score

wandb.save("saved/" + MODEL_NAME + ".h5")

log_cm("final", "test", (X_test, y_test))