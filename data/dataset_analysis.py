"""
Dataset Exploratory Data Analysis and Preprocessing Module.

Parses cropped dataset directories categorized by BCS (Body Condition Score),
extracts metadata (cow IDs, location prefixes, image IDs), validates labeling consistency
(e.g., cows assigned to conflicting BCS classes), and generates visual distributions.
"""

from collections import Counter
import os
import re
from typing import Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Directory paths
DATASET_ADDRESS: str = '../cropped_dataset'
SAVE_DATA_ADDRESS: str = 'meta_data'

# Supported image extension filter
IMAGE_EXTENSION: str = ".jpg"


def find_file_names_patterns() -> None:
    """
    Extract and display naming pattern frequencies across BCS folders.

    Replaces digit sequences in filenames with 'N' to identify structural conventions.
    """
    patterns: Counter = Counter()

    for bcs in ['3.25', '3.5', '3.75', '4.0', '4.25']:
        folder_path = os.path.join(DATASET_ADDRESS, bcs)

        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith('.jpg'):
                continue

            name = os.path.splitext(file_name)[0]

            # Replace digit sequences with 'N' to standardize patterns
            pattern = re.sub(r'\d+', 'N', name)
            patterns[pattern] += 1

    for pattern, count in patterns.most_common(50):
        print(f'{pattern:30} {count}')


def get_all_groups() -> pd.DataFrame:
    """
    Scan dataset folders, parse file metadata, and construct metadata DataFrame.

    Returns:
        pd.DataFrame: Structured metadata containing file names, BCS, prefixes,
                      cow IDs, image IDs, unique cow group IDs, and full paths.
    """
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


def number_of_unique_cows(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the count of distinct cow IDs per BCS category.

    Args:
        df (pd.DataFrame): Dataset metadata DataFrame.

    Returns:
        pd.Series: Unique cow counts indexed by BCS.
    """
    return df.groupby('bcs')["cow_id"].nunique()


def check_cow_ids_between_locations() -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Compare cow IDs across distinct farm locations (e.g., GS and YM).

    Returns:
        Tuple[Set[str], Set[str], Set[str]]:
            - gs_cows: Set of unique cow IDs from GS location.
            - ym_cows: Set of unique cow IDs from YM location.
            - common_cows: Set of shared cow IDs across both locations.
    """
    gs_cows = set(df[df['prefix'] == 'GS']['cow_id'].unique())
    ym_cows = set(df[df['prefix'] == 'YM']['cow_id'].unique())

    common_cows = gs_cows.intersection(ym_cows)
    return gs_cows, ym_cows, common_cows


def plot_bcs_distribution(df: pd.DataFrame) -> None:
    """
    Plot total image count distribution across BCS classes.

    Args:
        df (pd.DataFrame): Dataset metadata DataFrame.
    """
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


def plot_bcs_distribution_unique_cows(df: pd.DataFrame) -> None:
    """
    Plot unique cow subject count distribution across BCS classes.

    Args:
        df (pd.DataFrame): Dataset metadata DataFrame.
    """
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


def plot_location_distribution(df: pd.DataFrame) -> None:
    """
    Plot total image counts grouped by farm location prefix.

    Args:
        df (pd.DataFrame): Dataset metadata DataFrame.
    """
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


def plot_bcs_by_location(df: pd.DataFrame) -> None:
    """
    Plot BCS distributions segmented by location prefix.

    Args:
        df (pd.DataFrame): Dataset metadata DataFrame.
    """
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


def check_multiple_bcs_per_cow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify individual cow subjects assigned to multiple conflicting BCS classes.

    Args:
        df (pd.DataFrame): Dataset metadata DataFrame.

    Returns:
        pd.DataFrame: Summary table of cows associated with conflicting labels.
    """
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


def remove_cows_with_multiple_bcs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out all images belonging to cows labeled with multiple BCS values.

    Args:
        df (pd.DataFrame): Input metadata DataFrame.

    Returns:
        pd.DataFrame: Cleaned metadata DataFrame with consistent per-cow BCS labels.
    """
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

    # Identify cows assigned to two or more conflicting BCS classes
    problems = check_multiple_bcs_per_cow(df)
    print(f'number of cows in more than one bcs: {len(problems)}')
    print(f'examples: {problems.head()}')

    # Remove inconsistent subjects
    df = remove_cows_with_multiple_bcs(df)

    # Export clean metadata
    df.to_csv(f'{SAVE_DATA_ADDRESS}/all_images.csv', index=False)
    print(f"number of unique cows:\n {number_of_unique_cows(df)}\n")

    # Evaluate overlap between locations
    gs_cows, ym_cows, common_cows = check_cow_ids_between_locations()
    print(f'number of unique cows in GS: {len(gs_cows)}')
    print(f'number of unique cows in YM: {len(ym_cows)}')
    print(f'number common cow ids between locations: {len(common_cows)}\n')

    # Generate distribution plots
    plot_bcs_distribution(df)
    plot_bcs_distribution_unique_cows(df)
    plot_location_distribution(df)
    plot_bcs_by_location(df)
