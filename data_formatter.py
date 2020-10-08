import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

DIR_PATH = "archive"

testing_dir = "Testing_2"
training_dir = "Training_2"

path_full = "/outputFull"
path_small = "/outputSmall"

# Taken from: https://ieeexplore.ieee.org/document/1306572
training_set = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
testing_set = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


all_half_window_sizes = [90, 180, 360]

def read_annot(file_name):
    df = pd.read_fwf(file_name)

    # We ignore the Sub, chan, and num columns.
    df = df.drop(["Sub", "Chan", "Num", "Aux"], axis=1)

    # Replace annotations so they correspond to the five classes
    df = df.replace(["N", "L", "R"], 0)
    df = df.replace(["e", "j", "A", "a", "J", "S"], 1)
    df = df.replace(["V", "E"], 2)
    df = df.replace(["F"], 3)
    df = df.replace(["/", "f", "Q"], 4)

    # We now drop any rows with "Type" value not corresponding to the first four (we do not use class four, as in the paper) classes. (E.g., "+" (we are ignoring the Aux column) and "~").
    df = df.drop(df[(df.Type != 0) & (df.Type != 1) & (df.Type != 2) & (df.Type != 3)].index)

    return df


def read_csv_data(file_name):
    df = pd.read_csv(file_name)

    # Drop the last column, leaving only sample # and MLII
    df = df.drop(df.columns[-1], axis=1)

    # Rename for consistency
    df.columns = ["Sample #", "MLII"]

    return df


# Do it for training and testing
for current_dir, current_set in zip([training_dir, testing_dir], [training_set, testing_set]):

    # One frame for each window size we use
    total_frames = dict()

    for half_win_size in all_half_window_sizes:
        total_frames.update({half_win_size: pd.DataFrame()})


    for patient_num in current_set:
        # 1) ---------- Format the data, removing the columns and labels we do not use. ---------
        annot = read_annot("{}/{}annotations.txt".format(DIR_PATH, patient_num))
        csv = read_csv_data("{}/{}.csv".format(DIR_PATH, patient_num))

        merged_small = csv.merge(annot, how="inner", on="Sample #")
        merged_full = csv.merge(annot, how="left", on="Sample #")

        merged_full.to_csv("{}/{}output_full.csv".format(current_dir + path_full, patient_num))
        merged_small.to_csv("{}/{}output_small.csv".format(current_dir + path_small, patient_num))

        # 2) ---------- Find the baseline, and subtract it from the data: ---------
        data = merged_full.to_numpy()

        baseline = median_filter(data[:, 1].astype("float32"), size=73, mode="nearest")
        baseline = median_filter(baseline, size=217, mode="constant", cval=1000)

        merged_full["MLII"] = merged_full["MLII"] - baseline

        merged_full.to_csv("{}/{}output_full_baselined.csv".format(current_dir + path_full, patient_num))

        # 3) --------- "Segment" the data (taking windows around the annotation ---------
        data = merged_full.to_numpy()

        y = data[:, -1]
        X = data[:, :-1]

        y = y.astype("float32")
        X_only_labels = X[~np.isnan(y)]
        y_only_labels = y[~np.isnan(y)]

        # Indexes of labels:
        label_indexes = np.where(~np.isnan(y))[0]

        # For a window +/- half_window_size centred around the annotation
        for half_window_size in all_half_window_sizes:
            # Create new output array
            output = np.empty((0, half_window_size * 2 + 2))
            for pos, adj_idx in enumerate(label_indexes):

                segment = data[adj_idx - half_window_size: adj_idx + half_window_size + 1, 1]

                if segment.shape[0] >= half_window_size * 2 + 1:
                    # Check to ensure the first and last segments have enough data points
                    seg_with_label = np.empty(half_window_size * 2 + 2)

                    seg_with_label[:-1] = segment
                    seg_with_label[-1] = y_only_labels[pos]

                    output = np.concatenate((output, seg_with_label.reshape(1, -1)))

            output_df = pd.DataFrame(output)
            output_df.to_csv(
                "{}/{}output_full_baselined_split_{}.csv".format(current_dir + path_full, patient_num, half_window_size))


            total_frames[half_window_size] = total_frames[half_window_size].append(output_df)

        print("Processed patient", patient_num)

    # Save the combined frames
    for half_win_size in all_half_window_sizes:
        total_frames[half_win_size].to_csv("{}/combined_output_full_baselined_split_{}.csv".format(current_dir + path_full, half_win_size))
