import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
import os


image_path = []
labels = []

for label in os.listdir("dataset/train"):
    for image in os.listdir(f"dataset/train/{label}"):
        image_path.append(f"dataset/train/{label}/{image}")
        labels.append(label)

for label in os.listdir("dataset/val"):
    for image in os.listdir(f"dataset/val/{label}"):
        image_path.append(f"dataset/val/{label}/{image}")
        labels.append(label)

dataset_df = pd.DataFrame(zip(image_path, labels), columns=["image_path", "label"])
print(dataset_df.head())
print(dataset_df["label"].unique())

train_df = dataset_df.sample(frac=0.7)
test_df = dataset_df.drop(train_df.index)
validate_df = test_df.sample(frac=0.5)
test_df = test_df.drop(validate_df.index)

print(train_df.shape, test_df.shape, validate_df.shape)

label_encoder = LabelEncoder()

label_encoder.fit(dataset_df["label"])

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])


class CustomDataset(Dataset):
    def __init__(self, dataframe, base_dir="", transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.base_dir = base_dir
        self.labels = torch.tensor(label_encoder.transform(dataframe["label"]))

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, item):
        relative_path = self.dataframe.iloc[item]["image_path"]
        label = self.labels[item]
        image = Image.open(relative_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pooling = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()

        self.linear = nn.Linear((128*16*16), 128)

        self.output = nn.Linear(128, len(dataset_df["label"].unique()))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)

        return x


train_dataset = CustomDataset(dataframe=train_df, base_dir="dataset", transform=transform)
validate_dataset = CustomDataset(dataframe=validate_df, base_dir="dataset", transform=transform)
test_dataset = CustomDataset(dataframe=test_df, base_dir="dataset", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validate_loader = DataLoader(validate_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

LR = 1e-4
epochs = 10
model = NN()

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

total_loss_train_plot = []
total_loss_validate_plot = []
total_accuracy_train_plot = []
total_accuracy_validate_plot = []

for e in range(epochs):
    total_accuracy_train = 0
    total_accuracy_validation = 0
    total_loss_train = 0
    total_loss_validation = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_train = criterion(outputs, labels)
        total_loss_train += loss_train.item()

        loss_train.backward()

        accuracy_train = (torch.argmax(outputs, axis=1) == labels).sum().item()

        total_accuracy_train += accuracy_train
        optimizer.step()

    with torch.no_grad():
        for inputs, labels in validate_loader:
            outputs = model(inputs)
            validate_loss = criterion(outputs, labels)
            total_loss_validation += validate_loss.item()

            accuracy_validate = (torch.argmax(outputs, axis=1) == labels).sum().item()
            total_accuracy_validation += accuracy_validate

    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_loss_validate_plot.append(round(total_loss_validation/1000, 4))
    total_accuracy_train_plot.append(round(total_accuracy_train/train_dataset.__len__() * 100, 4))
    total_accuracy_validate_plot.append(round(total_accuracy_validation/validate_dataset.__len__() * 100, 4))

    print(f'''
    Epoch{e+1}/{epochs}, Train Loss: {round(total_loss_train/1000, 4)} Train Accuracy: {round((total_accuracy_train/train_dataset.__len__() * 100), 4)}
    Validation Loss: {round(total_loss_validation/1000, 4)} Validation Accuracy: {round(total_accuracy_validation/validate_dataset.__len__() * 100, 4)}
''')

    with torch.no_grad():
        total_loss_test = 0
        total_accuracy_test = 0
        for inputs, labels in test_loader:
            predictions = model(inputs)
            accuracy_test = (torch.argmax(predictions, axis=1) == labels).sum().item()
            total_accuracy_test += accuracy_test
            test_loss = criterion(predictions, labels)
            total_loss_test += test_loss.item()

    print(f"Accuracy is: {round((total_accuracy_test/test_dataset.__len__()) * 100, 4)} Loss is: {round(total_loss_test/1000, 4)}")

torch.save(model.state_dict(), "cnn.pth")

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

axs[0].plot(total_loss_train_plot, label='Training Loss')
axs[0].plot(total_loss_validate_plot, label='Validation Loss')
axs[0].set_title('Training and Validation Loss over Epochs')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

axs[1].plot(total_accuracy_train_plot, label='Training Accuracy')
axs[1].plot(total_accuracy_validate_plot, label='Validation Accuracy')
axs[1].set_title('Training and Validation Accuracy over Epochs')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

plt.show()




