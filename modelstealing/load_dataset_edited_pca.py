import os
import torch
import numpy as np
from taskdataset import TaskDataset
import random
from PIL import Image, ImageEnhance, ImageOps
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter, defaultdict

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

def pca_and_spectral_clustering(dataset, output_base_filename, image_size=(32, 32), cluster_counts=[5, 6, 7, 8, 9], n_components=50):
    images, ids, labels = dataset.imgs, dataset.ids, dataset.labels

    # Standardize images
    standardized_images = [img.resize((32, 32)).convert('RGB') for img in images]
    image_matrix = np.array([np.array(image).reshape(-1) for image in standardized_images])
    scaler = StandardScaler()
    image_matrix_scaled = scaler.fit_transform(image_matrix)

    # Apply Randomized PCA
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=random_seed)
    reduced_data = pca.fit_transform(image_matrix_scaled)
    
    for n_clusters in cluster_counts:
        print(f'Processing {n_clusters} clusters...')
        spectral_clustering = SpectralClustering(n_clusters=n_clusters, random_state=random_seed, assign_labels='discretize')
        original_cluster_labels = spectral_clustering.fit_predict(reduced_data)
        corrected_cluster_labels = correct_cluster_labels(original_cluster_labels, labels)

        # Count corrections
        corrections = sum(1 for original, corrected in zip(original_cluster_labels, corrected_cluster_labels) if original != corrected)
        print(f'Number of corrections for {n_clusters} clusters: {corrections}')

        # Save original and corrected cluster images
        save_dir_original = os.path.join(IMG_DIR, f'clusters_original_{n_clusters}_pca')
        save_dir_corrected = os.path.join(IMG_DIR, f'clusters_corrected_{n_clusters}_pca')
        os.makedirs(save_dir_original, exist_ok=True)
        os.makedirs(save_dir_corrected, exist_ok=True)
        save_cluster_images(images, original_cluster_labels, n_clusters, save_dir_original)
        save_cluster_images(images, corrected_cluster_labels, n_clusters, save_dir_corrected)

        # Write cluster info to CSV
        output_filename = f'{output_base_filename}_{n_clusters}_pca.csv'
        with open(output_filename, 'w') as f:
            f.write('id,original_cluster,corrected_cluster,label\n')
            for id, original_cluster, corrected_cluster, label in zip(ids, original_cluster_labels, corrected_cluster_labels, labels):
                f.write(f'{id},{original_cluster + 1},{corrected_cluster + 1},{label}\n')


def correct_cluster_labels(labels, original_labels):
    """
    Correct the cluster labels so that all images with the same label are in the same cluster.
    It assigns them to the cluster where the most images of that label are currently located.

    :param labels: List of assigned cluster labels.
    :param original_labels: List of original labels for the images.
    :return: Corrected list of cluster labels.
    """
    label_to_cluster = defaultdict(Counter)
    # Count how many times each label appears in each cluster
    for label, cluster in zip(original_labels, labels):
        label_to_cluster[label][cluster] += 1

    # Determine the most common cluster for each label
    most_common_cluster = {label: clusters.most_common(1)[0][0] for label, clusters in label_to_cluster.items()}

    # Correct the cluster assignments based on the most common cluster for each label
    corrected_labels = [most_common_cluster[label] for label in original_labels]

    return corrected_labels



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
    print("Images number:", len(dataset.ids))
    # print(type(dataset.imgs[0]))
    # save_pngs() # uncomment if images not present
    # visualize_random100(dataset.imgs)
    
    pca_and_spectral_clustering(dataset, os.path.join(IMG_DIR, "spectral_clustering_results"), cluster_counts=[5, 6, 7, 8, 9])