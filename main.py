import neural as nn
import layer as layers
import optimize as optim

from data_loader import MNISTDataLoader

# Set hyperparameters
input_size = 784 # 28 x 28 images flattened
hidden_size = 128
num_classes = 10 # Digits 0-9
batch_size = 32
num_epochs = 2
learning_rate = 0.001

train_images_path = "../data/train-images.idx3-ubyte"
train_labels_path = "../data/train-labels.idx1-ubyte"
test_images_path = "../data/t10k-images.idx3-ubyte"
test_labels_path = "../data/t10k-labels.idx1-ubyte"

# Initialize the MNIST training and testing datasets
train_dataset = MNISTDataLoader(train_images_path, train_labels_path, batch_size, shuffle=True)
test_dataset = MNISTDataLoader(test_images_path, test_labels_path, batch_size)

# Define the neural network
class MNISTNetwork(nn.NeuralNetwork):

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = layers.Linear(input_size, hidden_size)
        self.relu = layers.ReLU()
        self.fc2 = layers.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = MNISTNetwork(input_size, hidden_size, num_classes)
criterion = optim.CrossEntropyLoss(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Beginning training")
for epoch in range(num_epochs):
    for batch_index, (images, labels) in enumerate(train_dataset):

        outputs = model.forward(images)
        loss = criterion(outputs, labels)

        model.backward(labels)
        optimizer.step()

        if (batch_index + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1} / {num_epochs}] Loss: {loss}")

print("Beginning evaluation")
correct = 0
for images, labels in test_dataset:

    outputs = model.forward(images)
    predicted = [max(range(len(sample_pred)), key=sample_pred.__getitem__) for sample_pred in outputs]
    correct += sum(a == b for a, b in zip(predicted, labels))

print(f"Model Accuracy: {100 * (correct / test_dataset.num_samples)}")
