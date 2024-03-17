import matplotlib.pyplot as plt
import os
import random
from PIL import Image

# Set the directory where images are stored
directory = 'modelstealing/data/labels_ids_png/'

# Initialize variables
images_per_label = 5  # Five images for each label
total_labels = 50  # Total distinct labels

# Collecting images from the directory and sorting labels
label_images = {}
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        label = filename.split('_')[0]  # Extract label from filename
        if label not in label_images:
            label_images[label] = []
        label_images[label].append(os.path.join(directory, filename))

# Sort labels
sorted_labels = sorted(label_images.keys())[:total_labels]

# Prepare the plot
fig, axes = plt.subplots(nrows=10, ncols=25, figsize=(50, 40))  # 10 rows and 25 columns

# Flatten the 2D array of axes for easy iteration
axes = axes.flatten()

# Fill the grid with images, sorted by label
for i, label in enumerate(sorted_labels):
    # Get up to five images for the current label
    current_images = label_images[label][:images_per_label]
    for j, img_path in enumerate(current_images):
        ax_index = i * images_per_label + j  # Calculate the position in the grid
        img = Image.open(img_path)
        axes[ax_index].imshow(img)
        axes[ax_index].axis('off')  # Hide axis
        # Label only the third image of each label (in its third appearance)
        if j == 2:
            axes[ax_index].set_title(f"{label}", fontsize=10)

# Ensure any remaining axes are turned off if they're not used
for k in range(i * images_per_label + j + 1, len(axes)):
    axes[k].axis('off')

# Adjust the layout
plt.tight_layout()

# Adjust spacing between subplots: you can adjust these as needed
plt.subplots_adjust(hspace=0.3, wspace=0.1, top=0.95)  # Adjust top for space above

plt.show()