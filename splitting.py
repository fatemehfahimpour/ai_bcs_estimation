import os

import pandas as pd

SAVED_ADDRESS = 'meta_data'

METADATA_PATH = 'meta_data/all_images.csv'
OUTPUT_DIR = 'metadata/splits'

RANDOM_STATE = 42
TRAIN_SIZE = 0.60
VAL_SIZE = 0.25
TEST_SIZE = 0.15


def cow_meta_data(df):
    # image of same cows would be in a group
    cow_df = df.groupby(['cow_group_id']).agg(
        prefix=('prefix', 'first'),
        cow_id=('cow_id', 'first'),
        bcs=('bcs', 'first'),
        number_of_images=('file_name', 'count')
    ).reset_index()
    return cow_df




if __name__ == '__main__':
    df = pd.read_csv(METADATA_PATH)
    cow_df = cow_meta_data(df)
    print(cow_df.head(100))
