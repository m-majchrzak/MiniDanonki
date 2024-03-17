import pandas as pd

# Load the data
file_path = 'modelstealing/data/clusters_labels.txt'
data = pd.read_csv(file_path, sep='\t')

# Calculate value counts for each column
cluster_counts = data['cluster'].value_counts()
label_counts = data['label'].value_counts()

print(cluster_counts, label_counts)