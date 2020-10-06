import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# OUTPUT_PATH_FULL = "Training/outputFull"
OUTPUT_PATH_FULL = "Testing/outputFull"

# df = pd.read_csv("Training/outputFull/101output_full_baselined.csv")
#
# # Drop fist column that gets generated when saving as csv
# df = df.drop(df.columns[0], axis=1)
#
# data = df.to_numpy()

# Just taking the first 2000 for now
# data = data[:2000]
#
# y = data[:, -1]
# X = data[:, :-1]

# plt.figure(figsize=(16, 4))
# fig = plt.plot(X[:, 0], X[:, 1], c="black", zorder=1)

training_set = [101, 106, 108, 109, 112, 115, 118, 119, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
testing_set = [105, 111, 113, 121, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

differences = np.empty((1, 0))

for PATIENT_NUM in testing_set:
    df = pd.read_csv("{}/{}output_full_baselined.csv".format(OUTPUT_PATH_FULL, PATIENT_NUM))

    # Drop fist column that gets generated when saving as csv
    df = df.drop(df.columns[0], axis=1)

    data = df.to_numpy()

    y = data[:, -1]
    X = data[:, :-1]

    y = y.astype("float32")
    X_only_labels = X[~np.isnan(y)]
    y_only_labels = y[~np.isnan(y)]

    # Indexes of labels:
    label_indexes = np.where(~np.isnan(y))[0]

    # We adjust to actually be centred on the peak (in case the label wasn't exactly at the peak)
    # peak_search = 20  # Search 10 samples to the left and right of the annotation)
    # adjusted_indexes = np.empty(label_indexes.shape[0], dtype="uint32")
    # for r, idx in enumerate(label_indexes):
    #     data_slice = data[idx - peak_search:idx + peak_search + 1, 1]
    #     max_slice_index = np.argmax(data_slice)
    #     adjusted_indexes[r] = max_slice_index + idx - peak_search

    # We do not use the first and last item of the recording, as they might not have a full window size, as well as the baseline adjustment can affect them strangely.
    # adjusted_indexes = adjusted_indexes[1: -1]
    label_indexes = label_indexes[1: -1]
    y_only_labels = y_only_labels[1: -1]

    # Create new output array
    half_window_size = 90  # For a window +/- 90 centred around the peak
    output = np.empty((label_indexes.shape[0], half_window_size*2 + 2))
    for pos, adj_idx in enumerate(label_indexes):
        output[pos, :-1] = data[adj_idx - half_window_size: adj_idx + half_window_size + 1, 1]
        output[pos, -1] = y_only_labels[pos]

    output_df = pd.DataFrame(output)
    output_df.to_csv("{}/{}output_full_baselined_split.csv".format(OUTPUT_PATH_FULL, PATIENT_NUM))

    print(PATIENT_NUM)

# print("Mean difference for all training data:", differences.mean())

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
#
# print()