import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# OUTPUT_PATH_FULL = "Training/outputFull"
OUTPUT_PATH_FULL = "Testing/outputFull"
half_window_size = 90


training_set = [101, 106, 108, 109, 112, 115, 118, 119, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
testing_set = [105, 111, 113, 121, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

differences = np.empty((1, 0))

total_frame = pd.DataFrame()

for PATIENT_NUM in testing_set:
    df = pd.read_csv("{}/{}output_full_baselined_split_{}.csv".format(OUTPUT_PATH_FULL, PATIENT_NUM, half_window_size))

    # Drop fist column that gets generated when saving as csv
    df = df.drop(df.columns[0], axis=1)

    total_frame = total_frame.append(df)

    print(PATIENT_NUM)

print(total_frame.shape)
total_frame.to_csv("{}/combined_output_full_baselined_split_{}.csv".format(OUTPUT_PATH_FULL, half_window_size))

