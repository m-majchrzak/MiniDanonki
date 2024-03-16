import numpy as np
from PIL import Image
import os
import pandas as pd
import io
import umap
import matplotlib.pyplot as plt

# Wczytywanie zdjec
root_dir = 'data/datasets_preprocessed/all_filtered_train'

image_data = []
folders = []

for folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder)
    
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.png'):
                file_path = os.path.join(folder_path, file)
                
                img = Image.open(file_path).convert('L')
                img_array = np.array(img)

                flattened_img_array = img_array.flatten()

                image_data.append(flattened_img_array)
                folders.append(folder)


df = pd.DataFrame(image_data)
print(df)
raise Exception("...")


df.head()
df['Class'] = folders
df['Class'] = df["Class"].astype('int32')
df.dropna(inplace=True)

# UMAP
reducer = umap.UMAP(random_state=42)
reducer.fit(df.iloc[:, :-1])

mask = df['Class']  < 11

embedding = reducer.transform(df.iloc[:, :-1].loc[mask, :])

colors = df.loc[mask, 'Class']
size = len(set(colors))

plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(size+1)+1-0.5).set_ticks(np.arange(size)+1)