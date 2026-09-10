import os

import cv2
import numpy as np
import pandas as pd

METADATA_PATH = 'meta_data/all_images.csv'

DARK_PIXEL_THRESHOLD = 5
BRIGHT_PIXEL_THRESHOLD = 250
EXTREME_PIXEL_RATIO = 0.98

BLUR_THRESHOLD = 20


def check_image_readable(path):
    if not os.path.exists(path):
        return False

    image = cv2.imread(path)

    if image is None:
        return False

    return True


# is black or white?
def check_is_extreme_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_ration = np.mean(gray <= DARK_PIXEL_THRESHOLD)
    bright_ration = np.mean(gray >= BRIGHT_PIXEL_THRESHOLD)

    is_extreme = (dark_ration >= EXTREME_PIXEL_RATIO or bright_ration >= EXTREME_PIXEL_RATIO)
    return is_extreme, dark_ration, bright_ration


def check_is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    is_blurry = blur_score < BLUR_THRESHOLD
    return is_blurry, blur_score


def analyze_image(row):
    path = row['path']

    result = {
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


def analyze_dataset(df):
    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        result = analyze_image(row)
        results.append(result)

        if i % 100 == 0:
            print(f'analyzed {i} rows')

    return pd.DataFrame(results)


def remove_bad_images(df, quality_report):
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
