import os
from collections import Counter
import re

import pandas as pd

DATASET_ADDRESS = 'dataset'
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
                        'path': os.path.join(folder_path, file_name)
                    })

                else:

                    bad_file_format.append(file_name)

    print(f"bad file format number: {len(bad_file_format)}")
    print("Examples of bad filenames:")
    print(bad_file_format[:10])
    return pd.DataFrame(records)


def number_of_unique_cows(df):
    return df.groupby('bcs')["cow_id"].nunique()


if __name__ == '__main__':
    find_file_names_patterns()
    df = get_all_groups()
    print(number_of_unique_cows(df))
