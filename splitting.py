import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

SAVED_ADDRESS = 'meta_data'
METADATA_PATH = 'meta_data/all_images.csv'
OUTPUT_DIR = 'meta_data/splits'

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
    train_ratio = len(train_df) / len(df)
    val_ratio = len(val_df) / len(df)
    test_ratio = len(test_df) / len(df)

    size_error = abs(train_ratio - TRAIN_SIZE) + abs(test_ratio - TEST_SIZE) + abs(val_ratio - VAL_SIZE)
    total_bcs_distribution = (df['bcs'].value_counts(normalize=True).sort_index())

    def bcs_error(split_df):
        split_distribution = (
            split_df['bcs'].value_counts(normalize=True).reindex(total_bcs_distribution.index, fill_value=0))
        return np.abs(split_distribution - total_bcs_distribution).sum()

    bcs_error = bcs_error(train_df) + bcs_error(val_df) + bcs_error(test_df)
    return bcs_error + size_error


def split_cows(df, cow_df, iterations):
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

        error = calculate_split_score(df, train_df, val_df, test_df)
        if error < best_score:
            best_score = error
            best_split = (
                train_df.copy(),
                val_df.copy(),
                test_df.copy()
            )
    return best_split


def show_split_information(df, train_df, val_df, test_df):
    total_images = len(df)

    # Split ratios
    train_ratio = len(train_df) / total_images
    val_ratio = len(val_df) / total_images
    test_ratio = len(test_df) / total_images

    print('\n' + '=' * 70)
    print('SPLIT INFORMATION')
    print('=' * 70)

    print('\nImage distribution between splits:')
    print('-' * 70)

    print(
        f'Train:      {len(train_df):6d} images | '
        f'{train_ratio * 100:6.2f}% | '
        f'Target: {TRAIN_SIZE * 100:.2f}%'
    )

    print(
        f'Validation: {len(val_df):6d} images | '
        f'{val_ratio * 100:6.2f}% | '
        f'Target: {VAL_SIZE * 100:.2f}%'
    )

    print(
        f'Test:       {len(test_df):6d} images | '
        f'{test_ratio * 100:6.2f}% | '
        f'Target: {TEST_SIZE * 100:.2f}%'
    )

    # --------------------------------------------------
    # BCS distribution
    # --------------------------------------------------

    print('\n' + '=' * 70)
    print('BCS DISTRIBUTION')
    print('=' * 70)

    total_bcs = (
            df['bcs']
            .value_counts(normalize=True)
            .sort_index()
            * 100
    )

    train_bcs = (
            train_df['bcs']
            .value_counts(normalize=True)
            .reindex(total_bcs.index, fill_value=0)
            * 100
    )

    val_bcs = (
            val_df['bcs']
            .value_counts(normalize=True)
            .reindex(total_bcs.index, fill_value=0)
            * 100
    )

    test_bcs = (
            test_df['bcs']
            .value_counts(normalize=True)
            .reindex(total_bcs.index, fill_value=0)
            * 100
    )

    bcs_table = pd.DataFrame({
        'Total (%)': total_bcs,
        'Train (%)': train_bcs,
        'Validation (%)': val_bcs,
        'Test (%)': test_bcs
    })

    print('\nBCS percentage distribution:')
    print(bcs_table.round(2).to_string())

    # --------------------------------------------------
    # Number of cows
    # --------------------------------------------------

    print('\n' + '=' * 70)
    print('COW DISTRIBUTION')
    print('=' * 70)

    print(f'Total cows:      {df["cow_group_id"].nunique()}')
    print(f'Train cows:      {train_df["cow_group_id"].nunique()}')
    print(f'Validation cows: {val_df["cow_group_id"].nunique()}')
    print(f'Test cows:       {test_df["cow_group_id"].nunique()}')

    # --------------------------------------------------
    # Cow overlap check
    # --------------------------------------------------

    train_cows = set(train_df['cow_group_id'])
    val_cows = set(val_df['cow_group_id'])
    test_cows = set(test_df['cow_group_id'])

    train_val_overlap = train_cows & val_cows
    train_test_overlap = train_cows & test_cows
    val_test_overlap = val_cows & test_cows

    print('\n' + '=' * 70)
    print('COW OVERLAP CHECK')
    print('=' * 70)

    print(f'Train ∩ Validation: {len(train_val_overlap)}')
    print(f'Train ∩ Test:       {len(train_test_overlap)}')
    print(f'Validation ∩ Test:  {len(val_test_overlap)}')


def plot_bcs_distribution_comparison(df, train_df, val_df, test_df):
    bcs_order = sorted(df['bcs'].unique())

    distributions = {}

    datasets = {
        'Full Dataset': df,
        'Train': train_df,
        'Validation': val_df,
        'Test': test_df
    }

    for name, split_df in datasets.items():
        distribution = (
                split_df['bcs']
                .value_counts(normalize=True)
                .reindex(bcs_order, fill_value=0)
                * 100
        )

        distributions[name] = distribution

    distribution_df = pd.DataFrame(distributions)

    ax = distribution_df.plot(
        kind='bar',
        figsize=(10, 6)
    )

    plt.title('BCS Distribution Comparison')
    plt.xlabel('BCS')
    plt.ylabel('Percentage of Images (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Dataset')

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='%.1f%%',
            padding=3,
            fontsize=9
        )

    plt.tight_layout()
    plt.show()


def save_splits(train_df, val_df, test_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

    print('splits saved')



if __name__ == '__main__':
    df = pd.read_csv(METADATA_PATH)
    cow_df = cow_meta_data(df)

    train_df, val_df, test_df = split_cows(df, cow_df, 1000)
    plot_bcs_distribution_comparison(df, train_df, val_df, test_df)
    show_split_information(df, train_df, val_df, test_df)

    save_splits(train_df, val_df, test_df)
