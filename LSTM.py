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
from sklearn.metrics import confusion_matrix
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

# Updated callback to allow logging of confusion matrix
# class WandbClassificationCallback(WandbCallback):
#
#     def __init__(self, monitor='val_loss', verbose=0, mode='auto',
#                  save_weights_only=False, log_weights=False, log_gradients=False,
#                  save_model=True, training_data=None, validation_data=None,
#                  labels=[], data_type=None, predictions=1, generator=None,
#                  input_type=None, output_type=None, log_evaluation=False,
#                  validation_steps=None, class_colors=None, log_batch_frequency=None,
#                  log_best_prefix="best_",
#                  log_confusion_matrix=False,
#                  confusion_examples=0, confusion_classes=5):
#
#         super().__init__(monitor=monitor,
#                          verbose=verbose,
#                          mode=mode,
#                          save_weights_only=save_weights_only,
#                          log_weights=log_weights,
#                          log_gradients=log_gradients,
#                          save_model=save_model,
#                          training_data=training_data,
#                          validation_data=validation_data,
#                          labels=labels,
#                          data_type=data_type,
#                          predictions=predictions,
#                          generator=generator,
#                          input_type=input_type,
#                          output_type=output_type,
#                          log_evaluation=log_evaluation,
#                          validation_steps=validation_steps,
#                          class_colors=class_colors,
#                          log_batch_frequency=log_batch_frequency,
#                          log_best_prefix=log_best_prefix)
#
#         self.log_confusion_matrix = log_confusion_matrix
#         self.confusion_examples = confusion_examples
#         self.confusion_classes = confusion_classes
#
#     def on_epoch_end(self, epoch, logs={}):
#         if self.generator:
#             self.validation_data = next(self.generator)
#
#         if self.log_weights:
#             wandb.log(self._log_weights(), commit=False)
#
#         if self.log_gradients:
#             wandb.log(self._log_gradients(), commit=False)
#
#         if self.log_confusion_matrix:
#             if self.validation_data is None:
#                 wandb.termwarn(
#                     "No validation_data set, pass a generator to the callback.")
#             elif self.validation_data and len(self.validation_data) > 0:
#                 wandb.log(self._log_confusion_matrix(), commit=False)
#
#         if self.input_type in ("image", "images", "segmentation_mask") or self.output_type in (
#         "image", "images", "segmentation_mask"):
#             if self.validation_data is None:
#                 wandb.termwarn(
#                     "No validation_data set, pass a generator to the callback.")
#             elif self.validation_data and len(self.validation_data) > 0:
#                 if self.confusion_examples > 0:
#                     wandb.log({'confusion_examples': self._log_confusion_examples(
#                         confusion_classes=self.confusion_classes,
#                         max_confused_examples=self.confusion_examples)}, commit=False)
#                 if self.predictions > 0:
#                     wandb.log({"examples": self._log_images(
#                         num_images=self.predictions)}, commit=False)
#
#         wandb.log({'epoch': epoch}, commit=False)
#         wandb.log(logs, commit=True)
#
#         self.current = logs.get(self.monitor)
#         if self.current and self.monitor_op(self.current, self.best):
#             if self.log_best_prefix:
#                 wandb.run.summary["%s%s" % (self.log_best_prefix, self.monitor)] = self.current
#                 wandb.run.summary["%s%s" % (self.log_best_prefix, "epoch")] = epoch
#                 if self.verbose and not self.save_model:
#                     print('Epoch %05d: %s improved from %0.5f to %0.5f' % (
#                         epoch, self.monitor, self.best, self.current))
#             if self.save_model:
#                 self._save_model(epoch)
#             self.best = self.current
#
#     def _log_confusion_matrix(self):
#         x_val = self.validation_data[0]
#         y_val = self.validation_data[1]
#         y_val = y_val
#         y_pred = np.argmax(self.model.predict(x_val), axis=1)
#
#         confmatrix = confusion_matrix(y_pred, y_val, labels=range(len(self.labels)))
#         confdiag = np.eye(len(confmatrix)) * confmatrix
#         np.fill_diagonal(confmatrix, 0)
#
#         confmatrix = confmatrix.astype('float')
#         n_confused = np.sum(confmatrix)
#         confmatrix[confmatrix == 0] = np.nan
#         confmatrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': self.labels, 'y': self.labels, 'z': confmatrix,
#                                  'hoverongaps': False,
#                                  'hovertemplate': 'Predicted %{y}<br>Instead of %{x}<br>On %{z} examples<extra></extra>'})
#
#         confdiag = confdiag.astype('float')
#         n_right = np.sum(confdiag)
#         confdiag[confdiag == 0] = np.nan
#         confdiag = go.Heatmap({'coloraxis': 'coloraxis2', 'x': self.labels, 'y': self.labels, 'z': confdiag,
#                                'hoverongaps': False,
#                                'hovertemplate': 'Predicted %{y} just right<br>On %{z} examples<extra></extra>'})
#
#         fig = go.Figure((confdiag, confmatrix))
#         transparent = 'rgba(0, 0, 0, 0)'
#         n_total = n_right + n_confused
#         fig.update_layout({'coloraxis1': {'colorscale': [[0, transparent], [0, 'rgba(180, 0, 0, 0.05)'], [1,
#                                                                                                           f'rgba(180, 0, 0, {max(0.2, (n_confused / n_total) ** 0.5)})']],
#                                           'showscale': False}})
#         fig.update_layout({'coloraxis2': {
#             'colorscale': [[0, transparent], [0, f'rgba(0, 180, 0, {min(0.8, (n_right / n_total) ** 2)})'],
#                            [1, 'rgba(0, 180, 0, 1)']], 'showscale': False}})
#
#         xaxis = {'title': {'text': 'y_true'}, 'showticklabels': False}
#         yaxis = {'title': {'text': 'y_pred'}, 'showticklabels': False}
#
#         fig.update_layout(title={'text': 'Confusion matrix', 'x': 0.5}, paper_bgcolor=transparent,
#                           plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)
#
#         return {'confusion_matrix': wandb.data_types.Plotly(fig)}
#
#     def _log_confusion_examples(self, rescale=255, confusion_classes=5, max_confused_examples=3):
#         x_val = self.validation_data[0]
#         y_val = self.validation_data[1]
#         y_val = np.argmax(y_val, axis=1)
#         y_pred = np.argmax(self.model.predict(x_val), axis=1)
#
#         # Grayscale to rgb
#         if x_val.shape[-1] == 1:
#             x_val = np.concatenate((x_val, x_val, x_val), axis=-1)
#
#         confmatrix = confusion_matrix(y_pred, y_val, labels=range(len(self.labels)))
#         np.fill_diagonal(confmatrix, 0)
#
#         def example_image(class_index, x_val=x_val, y_pred=y_pred, y_val=y_val, labels=self.labels, rescale=rescale):
#             image = None
#             title_text = 'No example found'
#             color = 'red'
#
#             right_predicted_images = x_val[np.logical_and(y_pred == class_index, y_val == class_index)]
#             if len(right_predicted_images) > 0:
#                 image = rescale * right_predicted_images[0]
#                 title_text = 'Predicted right'
#                 color = 'rgb(46, 184, 46)'
#             else:
#                 ground_truth_images = x_val[y_val == class_index]
#                 if len(ground_truth_images) > 0:
#                     image = rescale * ground_truth_images[0]
#                     title_text = 'Example'
#                     color = 'rgb(255, 204, 0)'
#
#             return image, title_text, color
#
#         n_cols = max_confused_examples + 2
#         subplot_titles = [""] * n_cols
#         subplot_titles[-2:] = ["y_true", "y_pred"]
#         subplot_titles[max_confused_examples // 2] = "confused_predictions"
#
#         n_rows = min(len(confmatrix[confmatrix > 0]), confusion_classes)
#         fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subplot_titles)
#         for class_rank in range(1, n_rows + 1):
#             indx = np.argmax(confmatrix)
#             indx = np.unravel_index(indx, shape=confmatrix.shape)
#             if confmatrix[indx] == 0:
#                 break
#             confmatrix[indx] = 0
#
#             class_pred, class_true = indx[0], indx[1]
#             mask = np.logical_and(y_pred == class_pred, y_val == class_true)
#             confused_images = x_val[mask]
#
#             # Confused images
#             n_images_confused = min(max_confused_examples, len(confused_images))
#             for j in range(n_images_confused):
#                 fig.add_trace(go.Image(z=rescale * confused_images[j],
#                                        name=f'Predicted: {self.labels[class_pred]} | Instead of: {self.labels[class_true]}',
#                                        hoverinfo='name', hoverlabel={'namelength': -1}),
#                               row=class_rank, col=j + 1)
#                 fig.update_xaxes(showline=True, linewidth=5, linecolor='red', row=class_rank, col=j + 1, mirror=True)
#                 fig.update_yaxes(showline=True, linewidth=5, linecolor='red', row=class_rank, col=j + 1, mirror=True)
#
#             # Comparaison images
#             for i, class_index in enumerate((class_true, class_pred)):
#                 col = n_images_confused + i + 1
#                 image, title_text, color = example_image(class_index)
#                 fig.add_trace(
#                     go.Image(z=image, name=self.labels[class_index], hoverinfo='name', hoverlabel={'namelength': -1}),
#                     row=class_rank, col=col)
#                 fig.update_xaxes(showline=True, linewidth=5, linecolor=color, row=class_rank, col=col, mirror=True,
#                                  title_text=title_text)
#                 fig.update_yaxes(showline=True, linewidth=5, linecolor=color, row=class_rank, col=col, mirror=True,
#                                  title_text=self.labels[class_index])
#
#         fig.update_xaxes(showticklabels=False)
#         fig.update_yaxes(showticklabels=False)
#
#         return wandb.data_types.Plotly(fig)

hyperparameter_defaults = dict(
    scale_data = True
)

wandb.init(project="INFO411-Assignment2", notes="First tests, no balancing", config=hyperparameter_defaults)

# Initial setup
MODEL_NAME = "ECG_LSTM_Large_01"
SEQ_SIZE = 181
NUM_CLASSES = 4

df = pd.read_csv("Training/outputFull/combined_output_full_baselined_split_90.csv")

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
save_model = tf.keras.callbacks.ModelCheckpoint("saved/" + MODEL_NAME + ".h5", monitor='val_accuracy', verbose=0,
                                                save_best_only=True, save_weights_only=False, mode='max', period=1)

checkpoint_path = "checkpoints/" + MODEL_NAME + "cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=10)

def log_cm(tag, type, data):
    # print(val_gen.__getitem__(0)[1])
    X_in, y_in = data

    cm = confusion_matrix(
        y_in,
        np.argmax(model.predict(X_in), axis=1)
    )
    fig, ax = plt.subplots()
    ax.matshow(cm)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, z, ha='center', va='center')

    wandb.log({type + "CM_" + str(tag): fig})
    plt.savefig("CMs/" + type + "CM_" + str(tag) + ".png")

class Give_metrics(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            log_cm(epoch, "val", (X_val, y_val))
            log_cm(epoch, "train", (X_train, y_train))

opt = tf.keras.optimizers.Adam()

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile model
model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, validation_data=(X_val, y_val), epochs=50, callbacks=[WandbCallback(), save_model, cp_callback, Give_metrics()], shuffle=True)


# Evaluate on testing data
df = pd.read_csv("Testing/outputFull/combined_output_full_baselined_split_90.csv")

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

scores = model.evaluate(X_test, y_test)

for item, name in zip(scores, model.metrics_names):
    print('Test ' + name, item)
    wandb.run.summary['Test ' + name] = item

wandb.save("saved/" + MODEL_NAME + ".h5")

log_cm("final", "test", (X_test, y_test))