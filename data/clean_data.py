"""
Dataset Image Quality Assessment and Cleaning Pipeline.

Analyzes image readability, extreme pixel intensity distributions (near all-black
or all-white), and blurriness using Laplacian variance to filter out defective samples
from the dataset metadata.
"""

import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import pandas as pd

# Metadata CSV file path
METADATA_PATH: str = 'meta_data/all_images.csv'

# Thresholds for extreme intensity detection
DARK_PIXEL_THRESHOLD: int = 5
BRIGHT_PIXEL_THRESHOLD: int = 250
EXTREME_PIXEL_RATIO: float = 0.98

# Laplacian variance threshold for blur detection
BLUR_THRESHOLD: float = 20.0


def check_image_readable(path: str) -> bool:
    """
    Verify whether an image file exists and can be successfully decoded.

    Args:
        path (str): File system path to the image.

    Returns:
        bool: True if image exists and is readable, False otherwise.
    """
    if not os.path.exists(path):
        return False

    image = cv2.imread(path)

    if image is None:
        return False

    return True


def check_is_extreme_image(image: np.ndarray) -> Tuple[bool, float, float]:
    """
    Check if an image is predominantly black or white based on pixel ratios.

    Args:
        image (np.ndarray): Input BGR image.

    Returns:
        Tuple[bool, float, float]:
            - is_extreme (bool): True if extreme dark/bright ratio exceeds threshold.
            - dark_ratio (float): Fraction of pixels below dark threshold.
            - bright_ratio (float): Fraction of pixels above bright threshold.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_ratio = float(np.mean(gray <= DARK_PIXEL_THRESHOLD))
    bright_ratio = float(np.mean(gray >= BRIGHT_PIXEL_THRESHOLD))

    is_extreme = (dark_ratio >= EXTREME_PIXEL_RATIO or bright_ratio >= EXTREME_PIXEL_RATIO)
    return is_extreme, dark_ratio, bright_ratio


def check_is_blurry(image: np.ndarray) -> Tuple[bool, float]:
    """
    Estimate image blurriness using the variance of the Laplacian operator.

    Args:
        image (np.ndarray): Input BGR image.

    Returns:
        Tuple[bool, float]:
            - is_blurry (bool): True if variance score is below blur threshold.
            - blur_score (float): Calculated Laplacian variance.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    is_blurry = blur_score < BLUR_THRESHOLD
    return is_blurry, blur_score


def analyze_image(row: pd.Series) -> Dict[str, Any]:
    """
    Perform readability, intensity distribution, and blur analysis on a single image.

    Args:
        row (pd.Series): DataFrame row containing at least 'file_name' and 'path' keys.

    Returns:
        Dict[str, Any]: Detailed dictionary of quality metrics and rejection reasons.
    """
    path: str = row['path']

    result: Dict[str, Any] = {
        'file_name': row['file_name'],
        'path': path,

        'readable': False,

        'width': None,
        'height': None,

        'dark_ratio': None,
        'bright_ratio': None,

        'blur_score': None,

        'is_extreme': False,
        'is_blurry': False,

        'reason': 'OK'
    }

    if not check_image_readable(path):
        result['reason'] = 'unreadable'
        return result

    image = cv2.imread(path)
    result['readable'] = True

    is_extreme, dark_ratio, bright_ratio = check_is_extreme_image(image)
    result['dark_ratio'] = dark_ratio
    result['bright_ratio'] = bright_ratio
    result['is_extreme'] = is_extreme

    is_blurry, blur_score = check_is_blurry(image)
    result['blur_score'] = blur_score
    result['is_blurry'] = is_blurry

    if is_extreme:
        result['reason'] = 'almost_black_or_white'
    elif is_blurry:
        result['reason'] = 'very_blurry'

    return result


def analyze_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over the metadata DataFrame and evaluate quality metrics for all images.

    Args:
        df (pd.DataFrame): Input metadata DataFrame containing image paths.

    Returns:
        pd.DataFrame: Quality assessment report for each image.
    """
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        result = analyze_image(row)
        results.append(result)

        if i % 100 == 0:
            print(f'analyzed {i} rows')

    return pd.DataFrame(results)


def remove_bad_images(df: pd.DataFrame, quality_report: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out defective images based on readability, extreme intensity, and blurriness.

    Args:
        df (pd.DataFrame): Original metadata DataFrame.
        quality_report (pd.DataFrame): Quality analysis results.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - clean_df (pd.DataFrame): Cleaned metadata containing only valid images.
            - bad_images (pd.DataFrame): Sub-dataframe of rejected images with reasons.
    """
    bad_mask = ((~quality_report['readable']) | quality_report['is_extreme']
                | quality_report['is_blurry'])

    bad_images = quality_report[bad_mask]
    bad_paths = set(bad_images['path'])

    clean_df = df[~df['path'].isin(bad_paths)].copy()

    print('\nReasons for removal:')
    print(
        quality_report[bad_mask]['reason']
        .value_counts()
        .to_string()
    )

    return clean_df, bad_images


if __name__ == '__main__':
    df = pd.read_csv(METADATA_PATH)
    quality_report = analyze_dataset(df)

    clean_df, bad_images = remove_bad_images(df, quality_report)
    print(f'bad images number: {len(bad_images)}')
    clean_df.to_csv(METADATA_PATH, index=False)
