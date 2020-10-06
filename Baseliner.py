import numpy as np
from scipy.ndimage import median_filter
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_PATH_FULL = "Testing/outputFull"
# OUTPUT_PATH_SMALL = "Training/outputSmall"

# array = np.arange(10)
# np.random.shuffle(array)
# print(array)
#
# print(median_filter(array, size=3, mode="constant", cval=1000))

# df = pd.read_csv("Training/outputFull/101output_full.csv")
#
# # Drop fist column that gets generated when saving as csv
# df = df.drop(df.columns[0], axis=1)
#
# data = df.to_numpy()
#
# # Just taking the first 2000 for now
# data = data[:2000]


training_set = [101, 106, 108, 109, 112, 115, 118, 119, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
testing_set = [105, 111, 113, 121, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

for PATIENT_NUM in testing_set:
    df = pd.read_csv("{}/{}output_full.csv".format(OUTPUT_PATH_FULL, PATIENT_NUM))

    # Drop fist column that gets generated when saving as csv
    df = df.drop(df.columns[0], axis=1)

    data = df.to_numpy()

    baseline = median_filter(data[:, 1].astype("float32"), size=73, mode="nearest")
    baseline = median_filter(baseline, size=217, mode="constant", cval=1000)

    df["MLII"] = df["MLII"] - baseline

    df.to_csv("{}/{}output_full_baselined.csv".format(OUTPUT_PATH_FULL, PATIENT_NUM))

    print(PATIENT_NUM)



# baseline = median_filter(data[:, 1].astype("float32"), size=73, mode="nearest")
# baseline = median_filter(baseline, size=217, mode="constant", cval=1000)
#
# plt.figure(figsize=(16, 4))
# plt.plot(data[:, 0], data[:, 1], c="black", zorder=1)
# plt.plot(data[:, 0], baseline, c="green", zorder=1)
#
# plt.figure(figsize=(16, 4))
# plt.plot(data[:, 0], data[:, 1] - baseline, c="blue", zorder=1)
#
# plt.show()

