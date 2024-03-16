import os
import torch
import numpy as np
from taskdataset import TaskDataset
import random
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.cluster import SpectralClustering


# Set the random seed for all operations
random_seed = 1613
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


MAIN_DIR = "modelstealing/"
DATA_DIR = MAIN_DIR + "data/"
IMG_DIR = MAIN_DIR + "img/"


def save_pngs():
    DATA_SUBDIR = DATA_DIR + "ids_labels_png/"
    os.makedirs(DATA_SUBDIR, exist_ok=True)
    for img, id, label in zip(dataset.imgs, dataset.ids, dataset.labels):
        filename = f"{id}_{label}.png"
        filepath = os.path.join(DATA_SUBDIR, filename)
        img.save(filepath)

    DATA_SUBDIR = DATA_DIR + "ids_png/"
    os.makedirs(DATA_SUBDIR, exist_ok=True)
    for img, id in zip(dataset.imgs, dataset.ids):
        filename = f"{id}.png"
        filepath = os.path.join(DATA_SUBDIR, filename)
        img.save(filepath)

    DATA_SUBDIR = DATA_DIR + "labels_ids_png/"
    os.makedirs(DATA_SUBDIR, exist_ok=True)
    for img, id, label in zip(dataset.imgs, dataset.ids, dataset.labels):
        filename = f"{label}_{id}.png"
        filepath = os.path.join(DATA_SUBDIR, filename)
        img.save(filepath)

def visualize_random100(images):
    # Assuming images is a list of PIL.Image.Image objects
    random_images = random.sample(images, 100)  # Randomly select 100 images

    # Create a new image for the grid
    grid_width = 10
    grid_height = 10
    # Assuming all images are the same size, get the size of the first image
    image_width, image_height = random_images[0].size
    # Create a new empty image with the correct size
    grid_image = Image.new('RGB', (image_width * grid_width, image_height * grid_height))

    # Paste the images into the grid
    for index, image in enumerate(random_images):
        grid_x = (index % grid_width) * image_width
        grid_y = (index // grid_width) * image_height
        grid_image.paste(image, (grid_x, grid_y))

    # Display the grid image
    grid_image.save(IMG_DIR + "random100.png")
    grid_image.show()

def save_cluster_images(images, labels, cluster_number, save_dir, n=100, grid_size=(10, 10)):
    """
    Save a 10x10 grid of random images from a specified cluster.

    :param images: List of PIL.Image.Image objects.
    :param labels: Cluster labels for the images.
    :param cluster_number: The cluster number to visualize.
    :param save_dir: Directory where the image will be saved.
    :param n: Maximum number of images to include in the grid.
    :param grid_size: Dimensions of the grid (rows, columns).
    """
    # Filter images belonging to the specified cluster
    cluster_images = [img for img, label in zip(images, labels) if label == cluster_number]
    # Randomly select up to n images
    selected_images = random.sample(cluster_images, min(len(cluster_images), n))
    
    # Create the grid
    grid_image = Image.new('RGB', (grid_size[1] * selected_images[0].size[0], grid_size[0] * selected_images[0].size[1]))
    for index, image in enumerate(selected_images):
        grid_x = (index % grid_size[1]) * image.size[0]
        grid_y = (index // grid_size[1]) * image.size[1]
        grid_image.paste(image, (grid_x, grid_y))
    
    # Save the grid image
    grid_image.save(os.path.join(save_dir, f'cluster_{cluster_number + 1}.png'))

def umap_visualization_and_spectral_clustering(dataset, output_filename, image_size=(32, 32), n_clusters=6):
    images, ids, labels = dataset.imgs, dataset.ids, dataset.labels

    # Ensure all images are of the same size and format
    standardized_images = []
    for img in images:
        standardized_img = img.resize(image_size).convert('RGB')
        standardized_images.append(standardized_img)

    image_vectors = [np.array(image).flatten() for image in standardized_images]
    image_vectors = np.stack(image_vectors)

    # Apply UMAP to reduce the dimensionality to 2D
    umap_reducer = UMAP(random_state=random_seed)
    embedding = umap_reducer.fit_transform(image_vectors)

    # Apply Spectral Clustering on the reduced data
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, random_state=random_seed, assign_labels='discretize')
    cluster_labels = spectral_clustering.fit_predict(embedding)

    # Plot the results
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster_labels, cmap='Spectral', s=1)
    plt.title(f'UMAP projection of the dataset with {n_clusters} clusters', fontsize=24)
    plt.xlabel('UMAP dimension 1', fontsize=18)
    plt.ylabel('UMAP dimension 2', fontsize=18)
    plt.colorbar()
    plt.savefig(IMG_DIR + "umap_projection.png")

    # Saving images for each cluster and the mapping of ids to clusters and labels to a CSV file
    for cluster_number in range(n_clusters):
        save_cluster_images(images, cluster_labels, cluster_number, os.path.dirname(output_filename), n=100)

    with open(output_filename, 'w') as f:
        f.write('id,cluster,label\n')  # Header
        for id, cluster_label, label in zip(ids, cluster_labels, labels):
            f.write(f'{id},{cluster_label + 1},{label}\n')  # Adjusting cluster number for 1-based indexing

    return cluster_labels  # In case you want to use the labels later


def random_horizontal_flip(image: Image.Image, probability: float = 0.5) -> Image.Image:
    """
    Randomly flips the image horizontally.

    :param image: A PIL Image.
    :param probability: Probability of the image being flipped.
    :return: Horizontally flipped PIL Image.
    """
    if random.random() < probability:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def color_jitter(image: Image.Image, brightness: float = 0, contrast: float = 0, saturation: float = 0) -> Image.Image:
    """
    Randomly changes the brightness, contrast, and saturation of an image.

    :param image: A PIL Image.
    :param brightness: How much to jitter brightness (0 means no change).
    :param contrast: How much to jitter contrast (0 means no change).
    :param saturation: How much to jitter saturation (0 means no change).
    :return: A PIL Image with adjusted brightness, contrast, and saturation.
    """
    if brightness > 0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(max(0, 1 - brightness), 1 + brightness))
    
    if contrast > 0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(max(0, 1 - contrast), 1 + contrast))
    
    if saturation > 0:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.uniform(max(0, 1 - saturation), 1 + saturation))
    
    return image


def random_grayscale(image: Image.Image, probability: float = 0.1) -> Image.Image:
    """
    Randomly converts an image to grayscale.

    :param image: A PIL Image.
    :param probability: Probability of the image being converted to grayscale.
    :return: Grayscale PIL Image or original image.
    """
    if random.random() < probability:
        return ImageOps.grayscale(image)
    return image


def generate_augmentations(image: Image.Image):
    image_hf = random_horizontal_flip(image, 1) # horizontal flipped
    image_gr = random_grayscale(image, 1) # greyscale
    image_hf_gr = random_horizontal_flip(image_gr, 1)
    params_lists = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

    output_images = []
    for im in [image, image_hf, image_gr, image_hf_gr]:
        for params in params_lists:
            output_image = color_jitter(im, params[0], params[1], params[2])
            output_images.append(output_image)

    return output_images


if __name__ == "__main__":
    dataset = torch.load(DATA_DIR + "ModelStealingPub.pt")
    # print(dataset.ids, dataset.imgs, dataset.labels)
    # print(type(dataset.imgs[0]))
    save_pngs() # uncomment if images not present
    # print("Images number:", len(dataset.ids))
    # visualize_random100(dataset.imgs)
    
    umap_visualization_and_spectral_clustering(dataset, IMG_DIR + "umap_spectral_clustering_results.csv")