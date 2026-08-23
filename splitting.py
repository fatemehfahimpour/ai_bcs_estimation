import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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


def calculate_split_score(df, train_df, val_df, test_df):
    train_ratio = len(train_df) / len(train_df)
    val_ratio = len(val_df) / len(val_df)
    test_ratio = len(test_df) / len(test_df)

    size_error = abs(train_ratio - TRAIN_SIZE) + abs(test_ratio - TEST_SIZE) + abs(val_ratio - VAL_SIZE)
    total_bcs_distribution = (df['bcs'].value_counts(normalize=True).sort_index())

    def bcs_error(split_df):
        split_distribution = (
            split_df['bcs'].value_counts(normalize=True).reindex(total_bcs_distribution.index, fill_value=0))
        return np.abs(split_distribution - total_bcs_distribution).sum()

    bcs_error = bcs_error(train_df) + bcs_error(val_df) + bcs_error(test_df)
    return bcs_error + size_error


def split_cows(df, cos_df, iterations):
    best_score = float('inf')
    best_split = None

    for iteration in range(iterations):
        random_state = RANDOM_STATE + iteration

        train_cows, temp_cows = train_test_split(cow_df, test_size=VAL_SIZE + TEST_SIZE, random_state=random_state,
                                                 stratify=cow_df['bcs'])
        val_cows, test_cows = train_test_split(temp_cows, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
                                               stratify=temp_cows['bcs'], random_state=random_state)

        # finding ids in main df
        train_ids = set(train_cows['cow_group_id'])
        val_ids = set(val_cows['cow_group_id'])
        test_ids = set(test_cows['cow_group_id'])

        # finding all images of cows in main df
        train_df = df[df['cow_group_id'].isin(train_ids)]
        val_df = df[df['cow_group_id'].isin(val_ids)]
        test_df = df[df['cow_group_id'].isin(test_ids)]

        error = calculate_split_score(train_df, train_df, val_df, test_df)
        if error < best_score:
            best_score = error
            best_split = (
                train_df.copy(),
                val_df.copy(),
                test_df.copy()
            )

    return best_split


if __name__ == '__main__':
    df = pd.read_csv(METADATA_PATH)
    cow_df = cow_meta_data(df)
