import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Training/outputSmall/106output_small.csv")

# Drop fist column that gets generated when saving as csv
df = df.drop(df.columns[0], axis=1)

data = df.to_numpy()


y = data[:, -1]
X = data[:, :-1]

plt.figure(figsize=(16, 4))
fig = plt.plot(X[:, 0], X[:, 1], c="black", zorder=1)

plt.scatter(X[y==1, 0], X[y==1, 1], c="blue", marker="x", zorder=2, label="Class 1")
plt.scatter(X[y==2, 0], X[y==2, 1], c="green", marker="x", zorder=2, label="Class 2")
plt.scatter(X[y==3, 0], X[y==3, 1], c="purple", marker="x", zorder=2, label="Class 3")
plt.scatter(X[y==4, 0], X[y==4, 1], c="orange", marker="x", zorder=2, label="Class 4")

plt.legend(loc="best")

plt.gca().axes.get_xaxis().set_visible(False)
plt.show()

print()