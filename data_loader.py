import random
import struct
from array import array

class MNISTDataLoader:
    """
    Prepares the MNIST dataset for batch-wise iteration.

    Attributes:
        images_path: Relative path to the MNIST images binary file.
        labels_path: Relative path to the MNIST labels binary file.
        batch_size: Size of the batch returned at each iteration
        shuffle: Boolean to trigger randomizing the sample indices within each batch
        mean: The average pixel value of the dataset normalized between 0 and 1.
        std: The standard deviation of pixel values of the dataset normalized between 0 and 1.
    """

    def __init__(self, images_path: str, labels_path: str, batch_size: int, shuffle=False, mean=0.1307, std=0.3081):
        self.images_path = images_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mean = mean
        self.std = std

        # Open the MNIST images file to retrieve the numer of samples and image_size
        with open(self.images_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051: # Confirm the magic number matches the expected value
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        self.num_samples = size
        self.image_size = rows * cols

    def normalize(self, image_data: list[float]) -> list[float]:
        """
        Compute the in-place z-score normalization for a sample
        """
        for i, pixel in enumerate(image_data):
            scaled = pixel / 255 # Scale to [0, 1] range
            image_data[i] = (scaled - self.mean) / self.std # Standardize pixel value with mean and std

        return image_data

    def __iter__(self) -> iter:
        """
        Prepare the dataset for iteration by:
        - Opening the MNIST images and labels binary files
        - Preparing the batched indices for each iteration
        """
        self.images_file = open(self.images_path, "rb")
        self.labels_file = open(self.labels_path, "rb")

        indices = list(range(self.num_samples))
        if self.shuffle:
            random.shuffle(indices) # Randomize sample indices
        self.batches = [indices[i:i + self.batch_size] for i in range(0, self.num_samples, self.batch_size)]
        self.batch_index = 0

        return self

    def __next__(self) -> tuple[list[list[float]], list[int]]:
        """
        Return a batch of image data and corresponding labels.
        """
        if self.batch_index >= len(self.batches): # Handle halting iteration
            self.images_file.close()
            self.labels_file.close()
            raise StopIteration

        images = []
        labels = []
        for sample_index in self.batches[self.batch_index]: # Gather data for each sample in the batch
            image_offset = 16 + sample_index * self.image_size # Compute the byte-offset for the given image
            self.images_file.seek(image_offset)
            image_data = self.normalize(list(array("B", self.images_file.read(self.image_size))))
            images.append(image_data) # Append normalized image data to the batch

            label_offset = 8 + sample_index # Compute the byte-offset for the given label
            self.labels_file.seek(label_offset)
            labels.append(struct.unpack('B', self.labels_file.read(1))[0]) # Append the label to the batch

        self.batch_index += 1

        return images, labels
