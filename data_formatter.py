import numpy as np
import pandas as pd

DIR_PATH = "archive"
OUTPUT_PATH_FULL = "Testing/outputFull"
OUTPUT_PATH_SMALL = "Testing/outputSmall"

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

    # We now drop any rows with "Type" value not corresponding to the five classes. (E.g., "+" (we are ignoring the Aux column) and "~").
    df = df.drop(df[(df.Type != 0) & (df.Type != 1) & (df.Type != 2) & (df.Type != 3)].index)

    return df


def read_csv_data(file_name):
    df = pd.read_csv(file_name)

    # Drop the last column, leaving only sample # and MLII
    df = df.drop(df.columns[-1], axis=1)

    # Rename for consistency
    df.columns = ["Sample #", "MLII"]

    return df


training_set = [101, 106, 108, 109, 112, 115, 118, 119, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
testing_set = [105, 111, 113, 121, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231,232, 233, 234]

for PATIENT_NUM in testing_set:
    annot = read_annot("{}/{}annotations.txt".format(DIR_PATH, PATIENT_NUM))
    csv = read_csv_data("{}/{}.csv".format(DIR_PATH, PATIENT_NUM))

    merged_small = csv.merge(annot, how="inner", on="Sample #")
    merged_full = csv.merge(annot, how="left", on="Sample #")



    merged_full.to_csv("{}/{}output_full.csv".format(OUTPUT_PATH_FULL, PATIENT_NUM))
    merged_small.to_csv("{}/{}output_small.csv".format(OUTPUT_PATH_SMALL, PATIENT_NUM))
    print(PATIENT_NUM)