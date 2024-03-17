import os
import torch
import numpy as np
from taskdataset import TaskDataset
import random
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
from collections import Counter, defaultdict
from sklearn.decomposition import PCA

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


def calculate_purity(y_true, y_pred):
    # Calculate the purity, a measurement of how well the clustering reflects the true labels
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def evaluate_clustering(labels_true, labels_pred):
    # Calculate and return purity and adjusted Rand index
    purity = calculate_purity(labels_true, labels_pred)
    adjusted_rand = adjusted_rand_score(labels_true, labels_pred)
    return purity, adjusted_rand


def apply_pca_and_clustering(dataset, n_components=50, n_clusters_range=range(3, 10)):
    images = dataset.imgs
    ids = dataset.ids
    labels = dataset.labels

    # Ensure all images are of the same size and format
    standardized_images = []
    for img in images:
        standardized_img = img.resize((32, 32)).convert('RGB')  # Confirm size & convert to ensure consistency
        standardized_images.append(np.array(standardized_img).reshape(-1))  # Flatten the images

    # Stack images into a large matrix for PCA
    image_matrix = np.stack(standardized_images)

    # Apply PCA to reduce dimensions
    pca = PCA(n_components=n_components, random_state=random_seed)
    reduced_data = pca.fit_transform(image_matrix)
    
    # Perform clustering and evaluation for each method and number of clusters
    results = []
    for n_clusters in n_clusters_range:
        for ClusteringMethod in [KMeans, SpectralClustering]:
            clustering_model = ClusteringMethod(n_clusters=n_clusters, random_state=random_seed)
            labels_pred = clustering_model.fit_predict(reduced_data)
            purity, adjusted_rand = evaluate_clustering(labels, labels_pred)
            results.append({
                'method': ClusteringMethod.__name__,
                'n_clusters': n_clusters,
                'purity': purity,
                'adjusted_rand': adjusted_rand,
                'n_components': n_components
            })

    return results


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


def cluster_and_evaluate(embedding, labels_true, n_clusters_range=range(3, 10)):
    results = []
    for n_clusters in n_clusters_range:
        for ClusteringMethod in [KMeans, SpectralClustering]:
            # Initialize the clustering method
            if ClusteringMethod is KMeans:
                clustering_model = ClusteringMethod(n_clusters=n_clusters, random_state=random_seed)
            else:
                clustering_model = ClusteringMethod(n_clusters=n_clusters, random_state=random_seed, assign_labels='discretize')

            # Fit the model and predict clusters
            labels_pred = clustering_model.fit_predict(embedding)

            # Evaluate the clustering
            purity, adjusted_rand = evaluate_clustering(labels_true, labels_pred)
            
            # Save the results
            results.append({
                'method': ClusteringMethod.__name__,
                'n_clusters': n_clusters,
                'purity': purity,
                'adjusted_rand': adjusted_rand
            })
    
    # Return all results
    return results



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
    print("Images number:", len(dataset.ids))
    results = apply_pca_and_clustering(dataset)
    best_result = max(results, key=lambda x: x['purity'])  # or adjusted_rand for Adjusted Rand Index
    print(f"Best clustering method: {best_result['method']} with {best_result['n_clusters']} clusters, {best_result['n_components']} PCA components")
    print(f"Purity: {best_result['purity']}, Adjusted Rand Score: {best_result['adjusted_rand']}")