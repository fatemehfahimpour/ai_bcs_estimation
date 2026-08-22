import os
import pandas as pd

IMAGE_DIR = 'dataset'

df_3_25 = [f for f in os.listdir(f"{IMAGE_DIR}/3.25") if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
df_3_50 = [f for f in os.listdir(f"{IMAGE_DIR}/3.5") if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
df_3_75 = [f for f in os.listdir(f"{IMAGE_DIR}/3.75") if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
df_4_00 = [f for f in os.listdir(f"{IMAGE_DIR}/4.0") if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
df_4_25 = [f for f in os.listdir(f"{IMAGE_DIR}/4.25") if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

print(len(df_3_25))
print(len(df_3_50))
print(len(df_3_75))
print(len(df_4_00))
print(len(df_4_25))

