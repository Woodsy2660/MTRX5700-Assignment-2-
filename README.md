# Traffic Sign Classification with ResNet18

This project uses **ResNet18** (a classic deep convolutional neural network) to classify traffic sign images. The dataset is based on the [GTSRB (German Traffic Sign Recognition Benchmark)](https://benchmark.ini.rub.de/gtsrb_news.html), filtered to the following **5 classes**:

| New Label | Class Name | Original GTSRB Class ID |
|-----------|-----------|------------------------|
| 0 | Stop | 14 |
| 1 | Turn right | 33 |
| 2 | Turn left | 34 |
| 3 | Ahead only | 35 |
| 4 | Roundabout mandatory | 40 |

---

# Installation Guide

## 1. Python Environment

Python 3.8+ is required. We recommend using [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage your environment:

```bash
# Create a new environment (optional but recommended)
conda create -n traffic python=3.9
conda activate traffic
```

## 2. Install PyTorch

PyTorch is the deep learning framework used in this project. The CPU version is sufficient for this assignment; the GPU version will speed up training if you have an NVIDIA GPU.

### CPU Version (recommended for beginners)

Visit the [PyTorch Installation Page](https://pytorch.org/get-started/locally/), select your OS, choose `CPU` as the Compute Platform, and copy the install command.

Example (Linux / macOS):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### GPU Version (requires NVIDIA GPU)

If you have an NVIDIA GPU with CUDA installed, please contact your tutor for the correct installation command matching your CUDA version.

Verify GPU availability after installation:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 3. Install Additional Dependencies

```bash
pip install opencv-python pillow scipy numpy matplotlib pandas seaborn scikit-learn tqdm
```

> **What each library does:**
> - `numpy` — Array and matrix operations (PyTorch tensors and NumPy arrays are interconvertible)
> - `opencv-python`, `pillow` — Image reading and processing
> - `matplotlib`, `seaborn` — Plotting training curves, confusion matrices, and other visualizations
> - `pandas` — Data manipulation
> - `scikit-learn` — Provides `confusion_matrix`, `classification_report`, and other evaluation tools
> - `scipy` — Scientific computing utilities
> - `tqdm` — Displays progress bars during training

## 4. Verify Installation

Run the following to confirm all libraries are installed correctly:
```bash
python -c "import torch; import torchvision; import cv2; import numpy; import matplotlib; import sklearn; import tqdm; print('All dependencies installed successfully!')"
```

---

# Dataset Download

- [Google Drive Download Link](https://drive.google.com/drive/folders/1ZlGBDe9RKQqffznb6k4C6Px3ws5Sr_5A?usp=sharing)

After downloading, place `train.p`, `valid.p`, and `test.p` in the `Traffic-Sign-Student/` directory (next to the Python files):

```
Traffic-Sign-Student/
├── README.md
├── train_final.py          # Main training script (entry point)
├── network.py              # Neural network model definition
├── dataset.py              # Data loading and preprocessing
├── vis_utils.py            # Visualization utilities
├── train.p                 # Training data (download required)
├── valid.p                 # Validation data (download required)
├── test.p                  # Test data (download required)
├── results/                # Training results & visualizations (auto-generated)
│   ├── training_results.png
│   ├── predictions.png
│   └── confusion_matrix.png
└── checkpoint/             # Saved model weights (auto-generated)
    └── best_model.pth
```

> The data files are in Python pickle format (`.p`). Each file contains a dictionary with keys `'features'` (image arrays) and `'labels'` (label arrays).

---

# Code Structure and Concepts

## Overall Pipeline

```
Data Loading (dataset.py)  →  Model Definition (network.py)  →  Training Loop (train_final.py)  →  Evaluation & Visualization (vis_utils.py)
```

The full workflow is:
1. **Load data** — Read images from pickle files, filter to 5 traffic sign classes, split into train/validation/test sets
2. **Preprocess data** — Resize images, apply data augmentation, convert to tensors, normalize
3. **Define model** — Build the ResNet18 network architecture
4. **Train model** — Forward pass → compute loss → backward pass → update weights, repeat for many epochs
5. **Evaluate model** — Compute accuracy, confusion matrix, and classification report on the test set

---

## dataset.py — Data Loading & Preprocessing

This file transforms raw data into a format PyTorch can use.

### Key Concepts

- **Dataset** — A PyTorch abstract class representing a dataset. Must implement `__len__()` (returns dataset size) and `__getitem__()` (returns one sample by index).
- **DataLoader** — Wraps a Dataset into an iterable that yields batches. Supports shuffling, multi-threaded loading, etc.
- **Transform** — A pipeline of image transformations (resize, flip, normalize, etc.) applied to each image when it is loaded.

### Functions and Classes

| Function / Class | Purpose |
|-----------------|---------|
| `set_seed(seed)` | Sets random seeds for reproducibility across PyTorch and NumPy |
| `TrafficSignDataset` | PyTorch Dataset subclass that wraps images and labels |
| `TrafficSignDataset.__getitem__(idx)` | Returns one image and its label, applying transforms |
| `TrafficSignProcessor` | Main data processor — manages loading, filtering, and splitting |
| `TrafficSignProcessor.load_data(...)` | Loads pickle files, filters to target classes, remaps labels |
| `TrafficSignProcessor._filter_and_map_data(...)` | Filters target classes, remaps labels (e.g., original 14 → new 0), balances class counts |
| `TrafficSignProcessor.create_datasets(...)` | Creates train/valid/test Dataset objects with corresponding transforms |
| `TrafficSignProcessor.create_data_loaders(...)` | Wraps Datasets into DataLoaders with batch_size, shuffle, etc. |

### Image Normalization

The code normalizes images using **ImageNet mean and standard deviation**:
```python
mean = [0.485, 0.456, 0.406]  # per-channel mean (R, G, B)
std  = [0.229, 0.224, 0.225]  # per-channel std  (R, G, B)
```
Formula: `normalized_pixel = (pixel_value - mean) / std`

This stabilizes the input distribution and helps the model converge faster during training.

---

## network.py — Neural Network Model

This file defines the **ResNet18** architecture.

### Key Concepts

- **Conv2d (Convolutional Layer)** — Slides a small filter (kernel) across the image to extract local features (edges, textures, shapes).
- **BatchNorm2d (Batch Normalization)** — Normalizes each batch of features, which speeds up training and stabilizes gradients.
- **ReLU Activation** — `f(x) = max(0, x)`. Introduces non-linearity so the network can learn complex patterns.
- **Residual / Shortcut Connection** — The core idea of ResNet: the input is added directly to the output (`out += shortcut(x)`). This allows gradients to flow through the network more easily, enabling much deeper networks.

### ResNet18 Architecture Overview

```
Input Image (3, 32, 32)
    ↓
Conv2d (3→64) + BatchNorm + ReLU        # Initial convolution layer
    ↓
Layer1: 2 × BasicBlock (64→64)           # Residual layer 1
    ↓
Layer2: 2 × BasicBlock (64→128, stride=2) # Residual layer 2, spatial size halved
    ↓
Layer3: 2 × BasicBlock (128→256, stride=2) # Residual layer 3
    ↓
Layer4: 2 × BasicBlock (256→512, stride=2) # Residual layer 4
    ↓
Average Pooling (4×4)                     # Global average pooling
    ↓
Linear (512→5)                            # Fully connected layer, outputs scores for 5 classes
```

### Classes

| Class | Purpose |
|-------|---------|
| `BasicBlock` | The basic building block of ResNet: two 3×3 convolutions + a residual connection |
| `ResNet` | The full ResNet network, composed of stacked BasicBlocks |
| `ResNet18(num_classes=5)` | Factory function that creates a ResNet with 4 layers (2 blocks each) |

---

## train_final.py — Main Training Script

This is the **entry point** of the project. It handles model training, validation, and final evaluation.

### Key Concepts

- **Loss Function** — Measures the gap between model predictions and true labels. For classification, we typically use **CrossEntropyLoss**.
- **Optimizer** — Updates model parameters based on the loss gradient. Common choices: SGD, Adam, RMSprop.
- **Learning Rate Scheduler** — Dynamically adjusts the learning rate during training to help the model converge better.
- **Epoch** — One complete pass through the entire training dataset.
- **Forward Pass** — Input data flows through the network to produce predictions.
- **Backward Pass (Backpropagation)** — Gradients of the loss are computed with respect to each parameter, then the optimizer uses these gradients to update the parameters.

### Functions

| Function | Purpose |
|----------|---------|
| `train(model, train_loader, optimizer, criterion, epoch, epochs)` | Trains for one epoch: forward → loss → backward → update. Returns average loss and accuracy |
| `validate(model, val_loader, criterion)` | Evaluates the model on the validation set (no parameter updates). Returns average loss and accuracy |
| `save_checkpoint(...)` | Saves model weights and training state to a file, enabling training resumption |
| `main()` | Orchestrates everything: parse args → load data → create model → training loop → evaluate → visualize |

### Command-Line Arguments

```bash
python train_final.py [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 0.01 | Learning rate — controls how large each parameter update step is |
| `--batch_size` | 64 | Number of images per batch |
| `--epochs` | 100 | Number of training epochs |
| `--optimizer` | rmsprop | Optimizer choice: `sgd` / `adam` / `rmsprop` |
| `--scheduler` | cosine | Learning rate schedule: `cosine` / `step` / `none` |
| `--resume` / `-r` | — | Resume training from a saved checkpoint |
| `--weight_decay` | 5e-4 | Weight decay (L2 regularization to prevent overfitting) |
| `--momentum` | 0.9 | Momentum parameter for SGD optimizer |
| `--workers` | 1 | Number of data loading threads |

---

## vis_utils.py — Visualization Utilities

This file provides visualization functions to help you understand the data and evaluate model performance.

### Functions

| Function | Purpose | Output File |
|----------|---------|-------------|
| `visualize_class_examples(...)` | Displays sample images from each class | — |
| `visualize_image_intensity(...)` | Plots pixel intensity histograms | — |
| `visualize_before_after_preprocessing(...)` | Shows images before and after preprocessing side by side | — |
| `visualize_batch(...)` | Displays a batch of images from a DataLoader | — |
| `visualize_augmentations(...)` | Shows the effects of data augmentation on a single image | — |
| `visualize_dataset_statistics(...)` | Overview of dataset statistics (class distribution, image sizes, etc.) | — |
| `visualize_training_results(...)` | Plots training/validation loss and accuracy curves | `results/training_results.png` |
| `visualize_predictions(...)` | Shows model predictions on test images (green = correct, red = incorrect) | `results/predictions.png` |
| `plot_confusion_matrix(...)` | Generates a confusion matrix heatmap and classification report | `results/confusion_matrix.png` |
| `analyze_dataset(...)` | Prints detailed dataset statistics | — |

---

# TODO List — What You Need to Implement

There are **11 TODOs** across 4 files that you need to complete. They are listed below with detailed guidance (some closely related TODOs are grouped together).

---

## TODO 1 & 2: Configuration Parameters (dataset.py, lines 127–128)

```python
self.config = {
    ...
    'class_mapping': #TODO Find the class mapping,
    'img_size': #TODO Find the image size,
    ...
}
```

**What to do:**

- **`class_mapping`** — Provide a dictionary that maps original GTSRB class IDs to new labels 0–4. The original IDs are `[14, 33, 34, 35, 40]` and they should map to `[0, 1, 2, 3, 4]` respectively. Look at the `self.class_names` dictionary right below for reference.
- **`img_size`** — Set the input image size as an integer. Look at the `forward()` function in `network.py` — `F.avg_pool2d(out, 4)` tells you the expected spatial dimension before pooling, which gives you a hint about the input size (32).

---

## TODO 3: Data Augmentation (dataset.py, line 302)

```python
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    # Data Augmentation goes here
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

**What to do:**

Add data augmentation operations between `Resize` and `ToTensor`. Data augmentation artificially expands the training set by applying random transformations (rotation, color changes, etc.), which reduces overfitting.

**Useful `torchvision.transforms` operations:**

```python
transforms.RandomRotation(degrees=15)                       # Randomly rotate by up to ±15 degrees
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))    # Random translation
transforms.ColorJitter(brightness=0.2, contrast=0.2)        # Random brightness/contrast changes
transforms.RandomPerspective(distortion_scale=0.2, p=0.5)   # Random perspective distortion
```

> **Important:** Do NOT use `RandomHorizontalFlip` — flipping a "Turn left" sign would make it look like "Turn right", making the label incorrect!

---

## TODO 4: Build the ResNet Model (network.py, line 49)

```python
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=5):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # TODO Create the model according to the table
```

**What to do:**

Define all the layers that the `forward()` method uses. Look at `forward()` to see which attributes are needed:

| Attribute | Type | Description |
|-----------|------|-------------|
| `self.conv1` | `nn.Conv2d` | Initial convolution: 3 input channels → 64 output channels, kernel_size=3, stride=1, padding=1, bias=False |
| `self.bn1` | `nn.BatchNorm2d` | Batch normalization for conv1 output (64 features) |
| `self.layer1` | `self._make_layer(...)` | Layer 1: 64 channels, num_blocks[0] blocks, stride=1 |
| `self.layer2` | `self._make_layer(...)` | Layer 2: 128 channels, num_blocks[1] blocks, stride=2 |
| `self.layer3` | `self._make_layer(...)` | Layer 3: 256 channels, num_blocks[2] blocks, stride=2 |
| `self.layer4` | `self._make_layer(...)` | Layer 4: 512 channels, num_blocks[3] blocks, stride=2 |
| `self.linear` | `nn.Linear` | Fully connected layer: 512 × block.expansion inputs → num_classes outputs |

**Hint:** Read the `forward()` method line by line — every `self.xxx` used there must be defined in `__init__`.

---

## TODO 5 & 6: Calculate Loss and Accuracy (train_final.py, lines 73 and 112)

### In the `train()` function (line 73):

```python
# Available variables: train_loss, correct, total, loss, outputs, targets
# Statistics
# TODO calculate the loss and acc
```

### In the `validate()` function (line 112):

```python
# Available variables: val_loss, correct, total, loss, outputs, targets
# Statistics
# TODO calculate the loss and acc
```

**What to do (same logic for both):**

1. **Accumulate loss** — Add the current batch's loss to the running total
   - `loss.item()` extracts the loss as a plain Python number
   - Add it to `train_loss` (or `val_loss` in validate)
2. **Count correct predictions** — Find the predicted class from `outputs` and compare with `targets`
   - `outputs.max(1)` returns `(max_values, max_indices)` — the indices are the predicted classes
   - `predicted.eq(targets).sum().item()` counts how many predictions match the true labels
3. **Count total samples** — `targets.size(0)` gives the number of samples in the current batch
4. **Compute averages** — After the loop, calculate `avg_loss` and `acc`

**Hint:** Look at the `plot_confusion_matrix` function in `vis_utils.py` (lines 560–564) — it has nearly identical statistics-tracking code you can reference.

---

## TODO 7: Define Loss Function and Optimizer (train_final.py, lines 266 and 269)

```python
# Loss function
# TODO Define loss function

# Set optimizer
# TODO set the optimizer
```

**What to do:**

1. **Loss function** — Define a variable called `criterion` using `nn.CrossEntropyLoss()`. This is the standard loss function for multi-class classification.

2. **Optimizer** — Define a variable called `optimizer` based on the value of `args.optimizer`:
   - If `'sgd'` → `torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)`
   - If `'adam'` → `torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)`
   - If `'rmsprop'` → `torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)`

**Hint:** Both `criterion` and `optimizer` are used directly in the code that follows — `criterion` is passed to `train()` and `validate()`, and `optimizer` is passed to the training loop and checkpoint saving.

---

## TODO 8: Get Model Predictions (vis_utils.py, line 482)

```python
# Get predictions
# TODO get predcitions
```

**What to do:**

Generate predictions from the model for the given `inputs`:

1. Move `inputs` to the correct `device`
2. Wrap in `torch.no_grad()` (no gradient computation needed during inference)
3. Pass `inputs` through `model` to get outputs
4. Use `.argmax(1)` or `.max(1)` to get the predicted class indices, store in a variable called `preds`

**Hint:** The code below uses `preds[i].item()` to access individual predictions.

---

## TODO 9 & 10: Confusion Matrix and Classification Report (vis_utils.py, lines 577 and 582)

```python
# Calculate confusion matrix
# TODO create confusion matrix

# Print classification report
# TODO Implement classification report
```

**What to do:**

1. **Confusion matrix** — Use scikit-learn's `confusion_matrix(all_targets, all_preds)` and assign the result to a variable called `cm`
2. **Classification report** — Use scikit-learn's `classification_report(all_targets, all_preds, target_names=class_names_list)` and `print` the result

**Hint:** Both functions are already imported at the top of the file (`from sklearn.metrics import confusion_matrix, classification_report`). The variable `cm` is used in the code below to draw the heatmap.

---

# Training and Testing

## Basic Usage

After completing all TODOs, navigate to the student code directory and run:

```bash
cd Traffic-Sign-Student
python train_final.py
```

This single command will automatically:
1. Load and preprocess the data
2. Create the ResNet18 model
3. Train the model (default: 100 epochs)
4. Validate after each epoch
5. Save the best model to `checkpoint/`
6. Evaluate on the test set after training
7. Generate visualizations in `results/`

## Custom Parameters

```bash
# Use Adam optimizer with learning rate 0.001 for 50 epochs
python train_final.py --optimizer adam --lr 0.001 --epochs 50

# Use a larger batch size (if memory allows)
python train_final.py --batch_size 128

# Resume training from a saved checkpoint
python train_final.py --resume
```

## Expected Output

During training you will see output like:
```
Epoch 1/100 [Train]: 100%|████████| 20/20 [00:05<00:00] loss: 1.6032 | acc: 25.30%
Validating: 100%|████████████████| 5/5 [00:01<00:00]
Epoch 1 | Train Loss: 1.6032 | Train Acc: 25.30% | Val Loss: 1.5821 | Val Acc: 30.12%
Saving checkpoint...
```

After training completes, check the `results/` folder for visualizations:
- **training_results.png** — Training and validation loss/accuracy curves
- **predictions.png** — Sample model predictions on test images
- **confusion_matrix.png** — Classification confusion matrix heatmap

---

# FAQ

**Q: `FileNotFoundError: train.p`**
A: Make sure you downloaded the data files from Google Drive and placed them in the project root directory.

**Q: `ModuleNotFoundError: No module named 'xxx'`**
A: Run `pip install xxx` to install the missing library.

**Q: Training is very slow?**
A: CPU training is expected to be slower. You can reduce the number of epochs (`--epochs 20`) to verify your code is correct first, then run the full training afterwards.

**Q: Accuracy stays flat / does not improve?**
A: Double-check that the loss calculation and backpropagation in your TODOs are implemented correctly. You can also try adjusting the learning rate (`--lr 0.001`).
