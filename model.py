import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()

        self.linear = nn.Linear((128 * 16 * 16), 128)

        self.output = nn.Linear(128, 3)

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


def load_model():
    model = NN()
    model.load_state_dict(torch.load("cnn.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


def predict_image(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    output = model(img_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1).detach().numpy()[0]
    classes = ['cat', 'dog', 'wild']
    predicted_index = probabilities.argmax()
    predicted_label = classes[predicted_index]
    class_probability = {label: float(prob) for label, prob in zip(classes, probabilities)}

    return predicted_label, class_probability
