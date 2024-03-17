from modelstealing.our_dataset import OurDataset
from modelstealing.our_model.custom_cnn import CustomCNN
from modelstealing.our_model.model import TrainingParams

import torchvision.transforms as transforms

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Zmiana rozmiaru obrazu na 32x32
        transforms.ToTensor(),                       # Konwersja obrazu na tensor
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Konwersja obrazu na tensor
    ])
    dataset = OurDataset("1", transform=transform)

    model = CustomCNN()
    model.initialize_model()
    training_params = TrainingParams(num_epochs=100, learning_rate=0.0001, weight_decay=0.001, batch_size=32)
    model.train(dataset, training_params, "model1")
