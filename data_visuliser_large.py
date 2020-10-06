import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Training/outputFull/108output_full_baselined.csv")

# Drop fist column that gets generated when saving as csv
df = df.drop(df.columns[0], axis=1)

data = df.to_numpy()

# Just taking the first 2000 for now
n = 14223+250
data = data[n - 180: n + 180 + 1]

y = data[:, -1]
X = data[:, :-1]

plt.figure(figsize=(16, 4))
fig = plt.plot(X[:, 0], X[:, 1], c="black", zorder=1)

# Remove any nan rows to only leave us with ones with labels
y = y.astype("float32")
X_only_labels = X[~np.isnan(y)]
y_only_labels = y[~np.isnan(y)]

plt.scatter(X_only_labels[y_only_labels==0, 0], X_only_labels[y_only_labels==0, 1], c="red", marker="x", zorder=2, label="Class 0")
plt.scatter(X_only_labels[y_only_labels==1, 0], X_only_labels[y_only_labels==1, 1], c="blue", marker="x", zorder=2, label="Class 1")
plt.scatter(X_only_labels[y_only_labels==2, 0], X_only_labels[y_only_labels==2, 1], c="green", marker="x", zorder=2, label="Class 2")
plt.scatter(X_only_labels[y_only_labels==3, 0], X_only_labels[y_only_labels==3, 1], c="purple", marker="x", zorder=2, label="Class 3")
plt.scatter(X_only_labels[y_only_labels==4, 0], X_only_labels[y_only_labels==4, 1], c="orange", marker="x", zorder=2, label="Class 4")

plt.legend(loc="best")

plt.gca().axes.get_xaxis().set_visible(False)
plt.show()

print()