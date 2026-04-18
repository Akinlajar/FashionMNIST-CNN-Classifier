FashionMNIST CNN Classifier
A convolutional neural network built with PyTorch to classify grayscale images of clothing items from the FashionMNIST dataset into 10 categories.
Dataset
The project uses the FashionMNIST dataset, a drop-in replacement for the classic MNIST handwritten digits. It contains 70,000 grayscale images (28×28 pixels) across 10 clothing categories:
T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.
The data is split into 60,000 training images and 10,000 test images, loaded in batches of 64 using PyTorch's DataLoader.
Model Architecture
The CNN consists of three convolutional blocks followed by a fully connected output layer:
| Layer | Details |
|-------|---------|
| Conv Block 1 | Conv2d (1→16 channels, 3×3 kernel) → BatchNorm → ReLU → MaxPool (2×2) |
| Conv Block 2 | Conv2d (16→32 channels, 3×3 kernel) → BatchNorm → ReLU → MaxPool (2×2) |
| Conv Block 3 | Conv2d (32→64 channels, 3×3 kernel) → BatchNorm → ReLU → MaxPool (2×2) |
| Dropout | 50% dropout for regularization |
| Output | Fully connected layer (576 → 10 classes) |
Training Configuration
Optimizer: Adam (learning rate = 0.0001)
Loss Function: CrossEntropyLoss
Epochs: 20
Batch Size: 64
Results
The model achieves 90.25% accuracy on the test set after 20 epochs of training. Training accuracy steadily improved from ~70% in epoch 1 to ~91% by epoch 20, with training loss declining from 0.877 to 0.254.
Per-Class Accuracy
| Class | Accuracy |
|-------|----------|
| T-shirt/top | 88.2% |
| Trouser | 97.8% |
| Pullover | 90.0% |
| Dress | 89.3% |
| Coat | 83.8% |
| Sandal | 97.4% |
| Shirt | 64.9% |
| Sneaker | 97.7% |
| Bag | 97.7% |
| Ankle boot | 95.7% |
The model performs best on Trouser, Sneaker, and Bag categories, while Shirt is the most challenging class — often confused with T-shirt/top and Coat, as shown in the confusion matrix.
Visualizations
The notebook produces several plots: sample images from the dataset, a training loss curve across epochs, a grid of test predictions with correct/incorrect labels highlighted in green/red, and a confusion matrix heatmap.
Dependencies
Python 3.x
PyTorch & torchvision
matplotlib
scikit-learn
seaborn
Usage
Open the notebook and run all cells sequentially. The FashionMNIST dataset must be available locally at the path specified in the data loading cell, or you can set download=True to fetch it automatically.
