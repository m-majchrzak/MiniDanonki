import torch
from taskdataset import TaskDataset


if __name__ == "__main__":
    dataset = torch.load("modelstealing/data/ModelStealingPub.pt")

    #N = len(dataset.ids)
    N = 5

    for img_number in range(N):
        print(dataset.ids[img_number], dataset.imgs[img_number], dataset.labels[img_number])