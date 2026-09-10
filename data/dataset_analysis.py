import os
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DATASET_ADDRESS = '../cropped_dataset'
SAVE_DATA_ADDRESS = 'meta_data'

# IMAGE_EXTENSION = (".jpg", ".jpeg", ".png", ".bmp")
IMAGE_EXTENSION = (".jpg")


def find_file_names_patterns():
    patterns = Counter()

    for bcs in ['3.25', '3.5', '3.75', '4.0', '4.25']:

        folder_path = os.path.join(DATASET_ADDRESS, bcs)

        for file_name in os.listdir(folder_path):

            if not file_name.lower().endswith('.jpg'):
                continue

            name = os.path.splitext(file_name)[0]

            # replace numbers with N
            pattern = re.sub(r'\d+', 'N', name)

            patterns[pattern] += 1

    for pattern, count in patterns.most_common(50):
        print(f'{pattern:30} {count}')


def get_all_groups():
    records = []
    bcs_values = ['3.25', '3.5', '3.75', '4.0', '4.25']

    bad_file_format = []

    for bcs in bcs_values:
        folder_path = os.path.join(DATASET_ADDRESS, bcs)
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(IMAGE_EXTENSION):
                name = os.path.splitext(file_name)[0]
                parts = name.split('_')
                if len(parts) >= 3:
                    records.append({
                        'file_name': file_name,
                        'bcs': bcs,
                        'prefix': parts[0],
                        'cow_id': parts[1],
                        'image_id': parts[2],
                        'cow_group_id': f'{parts[0]}_{parts[1]}',
                        'path': os.path.join(folder_path, file_name)
                    })

                else:

                    bad_file_format.append(file_name)

    print(f"number of images with no cow id: {len(bad_file_format)}\n")
    return pd.DataFrame(records)


def number_of_unique_cows(df):
    return df.groupby('bcs')["cow_id"].nunique()


def check_cow_ids_between_locations():
    gs_cows = set(df[df['prefix'] == 'GS']['cow_id'].unique())
    ym_cows = set(df[df['prefix'] == 'YM']['cow_id'].unique())

    common_cows = gs_cows.intersection(ym_cows)
    return gs_cows, ym_cows, common_cows


def plot_bcs_distribution(df):
    # number of images for each bcs
    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x='bcs',
        order=sorted(df['bcs'].unique())
    )

    plt.title('Image Distribution by BCS')
    plt.xlabel('BCS')
    plt.ylabel('Number of Images')

    plt.tight_layout()
    plt.show()


def plot_bcs_distribution_unique_cows(df):
    cow_counts = (
        df.groupby('bcs')['cow_group_id']
        .nunique()
        .reset_index(name='number_of_cows')
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=cow_counts,
        x='bcs',
        y='number_of_cows',
        order=sorted(cow_counts['bcs'].unique())
    )

    plt.title('Cow Distribution by BCS')
    plt.xlabel('BCS')
    plt.ylabel('Number of Unique Cows')

    plt.tight_layout()
    plt.show()


def plot_location_distribution(df):
    # Number of images and cows in each location.
    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=df,
        x='prefix'
    )

    plt.title('Image Distribution by Location')
    plt.xlabel('Location')
    plt.ylabel('Number of Images')

    plt.tight_layout()
    plt.show()


def plot_bcs_by_location(df):
    # BCS distribution separately for GS and YM.

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x='bcs',
        hue='prefix',
        order=sorted(df['bcs'].unique())
    )

    plt.title('BCS Distribution by Location')
    plt.xlabel('BCS')
    plt.ylabel('Number of Images')
    plt.legend(title='Location')

    plt.tight_layout()
    plt.show()


def check_multiple_bcs_per_cow(df):
    problems = []

    for cow_id, group in df.groupby('cow_group_id'):
        bcs_counts = group['bcs'].value_counts().sort_index()
        if len(bcs_counts) > 1:
            problems.append({
                'cow_group_id': cow_id,
                'number_of_bcs': len(bcs_counts),
                'bcs_values': list(bcs_counts.index),
                'images_per_bcs': bcs_counts.to_dict(),
                'total_images': len(group)
            })

    return pd.DataFrame(problems)


def remove_cows_with_multiple_bcs(df):
    bcs_per_cow = df.groupby('cow_group_id')['bcs'].nunique()
    problematic_cows = bcs_per_cow[bcs_per_cow > 1].index

    cleaned_df = df[~df['cow_group_id'].isin(problematic_cows)].copy()

    print(f'Images before removing multiple bcs cows: {len(df)}')
    print(f'Images after removing multiple bcs cows: {len(cleaned_df)}\n')

    return cleaned_df


if __name__ == '__main__':
    print('name patterns: ')
    find_file_names_patterns()

    df = get_all_groups()
    # cows in 2 or more different bcs groups
    problems = check_multiple_bcs_per_cow(df)
    print(f'number of cows in more than one bcs: {len(problems)}')
    print(f'examples: {problems.head()}')
    # removing cows with more than 1 bcs
    df = remove_cows_with_multiple_bcs(df)
    # save
    df.to_csv(f'{SAVE_DATA_ADDRESS}/all_images.csv', index=False)
    print(f"number of unique cows:\n {number_of_unique_cows(df)}\n")

    # number of unique cows and common cows
    gs_cows, ym_cows, common_cows = check_cow_ids_between_locations()
    print(f'number of unique cows in GS: {len(gs_cows)}')
    print(f'number of unique cows in YM: {len(ym_cows)}')
    print(f'number common cow ids between locations: {len(common_cows)}\n')

    # plots
    plot_bcs_distribution(df)
    plot_bcs_distribution_unique_cows(df)
    plot_location_distribution(df)
    plot_bcs_by_location(df)
