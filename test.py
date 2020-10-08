import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report


# df = pd.read_csv("Training/outputFull/106output_full_baselined_split.csv")
#
# # Drop fist column that gets generated when saving as csv
# df = df.drop(df.columns[0], axis=1)
#
# data = df.to_numpy()
#
# # Just taking the first 2000 for now
#
# y = data[:, -1]
# X = data[:, :-1]
#
# plt.figure(figsize=(16, 4))
# fig = plt.plot(X[:, 0], X[:, 1], c="black", zorder=1)
#
# # Remove any nan rows to only leave us with ones with labels
# y = y.astype("float32")
# X_only_labels = X[~np.isnan(y)]
# y_only_labels = y[~np.isnan(y)]
#
# plt.scatter(X_only_labels[y_only_labels==0, 0], X_only_labels[y_only_labels==0, 1], c="red", marker="x", zorder=2, label="Class 0")
# plt.scatter(X_only_labels[y_only_labels==1, 0], X_only_labels[y_only_labels==1, 1], c="blue", marker="x", zorder=2, label="Class 1")
# plt.scatter(X_only_labels[y_only_labels==2, 0], X_only_labels[y_only_labels==2, 1], c="green", marker="x", zorder=2, label="Class 2")
# plt.scatter(X_only_labels[y_only_labels==3, 0], X_only_labels[y_only_labels==3, 1], c="purple", marker="x", zorder=2, label="Class 3")
# plt.scatter(X_only_labels[y_only_labels==4, 0], X_only_labels[y_only_labels==4, 1], c="orange", marker="x", zorder=2, label="Class 4")
#
# plt.legend(loc="best")
#
# plt.gca().axes.get_xaxis().set_visible(False)
# plt.show()


def calc_jk(true_arr, pred_arr):
    k_index = cohen_kappa_score(true_arr, pred_arr)

    cr = classification_report(true_arr, pred_arr, output_dict=True)

    print(cr)

    s_sensitivity = cr['1']['recall']
    v_sensitivity = cr['2']['recall']

    s_ppv = cr['1']['precision']
    v_ppv = cr['2']['precision']

    j_index = s_sensitivity + v_sensitivity + s_ppv + v_ppv

    jk_index = (1 / 2) * k_index + (1 / 8) * j_index
    return k_index, j_index, jk_index



def create_arrays(df):
    # Unstack to make tuples of actual,pred,count
    df = df.unstack().reset_index()

    # Pull the value labels and counts
    actual = df['Actual'].values
    predicted = df['Predicted'].values
    totals = df.iloc[:,2].values

    # Use list comprehension to create original arrays
    y_true = [[curr_val]*n for (curr_val, n) in zip(actual, totals)]
    y_predicted = [[curr_val]*n for (curr_val, n) in zip(predicted, totals)]

    # They come nested so flatten them
    y_true = [item for sublist in y_true for item in sublist]
    y_predicted = [item for sublist in y_predicted for item in sublist]

    return y_true, y_predicted

dataf = pd.DataFrame([[42244, 1540, 99, 150], [427, 1601, 21, 1], [90, 75, 3051, 4], [256, 2, 82, 48]])

dataf.index.name = 'Actual'
dataf.columns.name = 'Predicted'

t, p = create_arrays(dataf)

cm = confusion_matrix(t, p)
print(cm)


print(calc_jk(t, p))