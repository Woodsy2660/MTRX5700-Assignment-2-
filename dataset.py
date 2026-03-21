#!/usr/bin/env python
# coding: utf-8

"""
Traffic Sign Dataset Handling
-------------------------------------------------
Classes and utilities for loading, processing, and preparing traffic sign datasets
"""

import pickle
import numpy as np
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    """
    Set random seed to ensure experiment reproducibility
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set: {seed}")


class TrafficSignDataset(Dataset):
    """
    Traffic Sign Dataset Class
    """
    def __init__(self, features, labels, transform=None, orig_transform=None, class_mapping=None):
        """
        Initialize dataset
        
        Args:
            features (numpy.ndarray): Image data
            labels (numpy.ndarray): Labels
            transform (callable, optional): Data transformer (for returning preprocessed images)
            orig_transform (callable, optional): Original image transformer (if returning original images with basic processing)
            class_mapping (dict, optional): Class label mapping
        """
        self.features = features
        self.labels = labels
        self.transform = transform
        self.orig_transform = orig_transform
        
        # Label name mapping
        self.class_names = {
            0: 'Stop',
            1: 'Turn right',
            2: 'Turn left',
            3: 'Ahead only',
            4: 'Roundabout mandatory'
        }
        
        # Update class mapping (if provided)
        if class_mapping:
            self.class_names.update(class_mapping)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.features[idx]
        label = self.labels[idx]
        
        # If image is not RGB format, convert to RGB (3 channels)
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=2)
        
        # Convert to PIL image for applying transformations
        image = Image.fromarray(image.astype(np.uint8))
        
        # Save original image (may apply some basic transformations)
        orig_image = image
        if self.orig_transform:
            orig_image = self.orig_transform(image)
        
        # Apply preprocessing transformations
        if self.transform:
            image = self.transform(image)
        
        return (image, orig_image, label) if self.orig_transform else (image, label)
    
    def get_class_name(self, label):
        """
        Get class name
        
        Args:
            label (int): Class label
            
        Returns:
            str: Class name
        """
        return self.class_names.get(label, f"Unknown ({label})")


class TrafficSignProcessor:
    """
    Traffic Sign Data Processor: Responsible for data loading, filtering, analysis and visualization
    """
    def __init__(self, config=None):
        """
        Initialize data processor
        
        Args:
            config (dict, optional): Configuration parameters
        """
        # Default configuration
        self.config = {
            'balance_data': True,
            'max_samples_per_class': 300,
            'valid_classes': [14, 33, 34, 35, 40],
            'class_mapping': {14:0, 33:1, 34:2, 35:3, 40:4},
            'img_size': 32,
            'mean': [0.485, 0.456, 0.406],  # ImageNet mean
            'std': [0.229, 0.224, 0.225]    # ImageNet std
        }
        
        # Update configuration (if provided)
        if config:
            self.config.update(config)
        
        # Class names
        self.class_names = {
            0: 'Stop',
            1: 'Turn right',
            2: 'Turn left',
            3: 'Ahead only',
            4: 'Roundabout mandatory'
        }
        
        # Store loaded data
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
        
    def load_data(self, training_file, validation_file=None, testing_file=None, balance_data=None, max_samples_per_class=None):
        """
        Load and preprocess traffic sign dataset
        
        Args:
            training_file (str): Path to the training data pickle file
            validation_file (str, optional): Path to the validation data pickle file
            testing_file (str, optional): Path to the testing data pickle file
            balance_data (bool, optional): Whether to balance training data
            max_samples_per_class (int, optional): Maximum samples per class when balancing
            
        Returns:
            tuple: Preprocessed training, validation and testing data and labels
        """
        if balance_data is None:
            balance_data = self.config['balance_data']
        if max_samples_per_class is None:
            max_samples_per_class = self.config['max_samples_per_class']
        
        print(f"Loading data files...")
        
        # Load training data
        try:
            with open(training_file, mode='rb') as f:
                train = pickle.load(f)
            X_train_raw, y_train_raw = train['features'], train['labels']
            
            # If validation and test files are not provided, automatically split the training set
            if validation_file is None or testing_file is None:
                # First split out the test set
                X_train_temp, X_test_raw, y_train_temp, y_test_raw = train_test_split(
                    X_train_raw, y_train_raw, test_size=0.2, random_state=42, stratify=y_train_raw)
                
                # Split out validation set from remaining data
                X_train_raw, X_valid_raw, y_train_raw, y_valid_raw = train_test_split(
                    X_train_temp, y_train_temp, test_size=0.25, random_state=42, stratify=y_train_temp)
                
                print(f"Data automatically split into training set ({len(X_train_raw)} samples), validation set ({len(X_valid_raw)} samples) and test set ({len(X_test_raw)} samples)")
            
            # If validation and test files are provided, load them
            else:
                if validation_file:
                    with open(validation_file, mode='rb') as f:
                        valid = pickle.load(f)
                    X_valid_raw, y_valid_raw = valid['features'], valid['labels']
                
                if testing_file:
                    with open(testing_file, mode='rb') as f:
                        test = pickle.load(f)
                    X_test_raw, y_test_raw = test['features'], test['labels']
        
        except Exception as e:
            print(f"Error loading data files: {e}")
            raise
        
        print("Data loading complete. Filtering and mapping data...")
        
        # Filter and map data
        X_filtered_train, y_filtered_train = self._filter_and_map_data(
            X_train_raw, y_train_raw, balance=balance_data, max_count=max_samples_per_class
        )
        
        X_filtered_valid, y_filtered_valid = self._filter_and_map_data(
            X_valid_raw, y_valid_raw, balance=False
        )
        
        X_filtered_test, y_filtered_test = self._filter_and_map_data(
            X_test_raw, y_test_raw, balance=False
        )
        
        # Store preprocessed data
        self.X_train = X_filtered_train
        self.y_train = y_filtered_train
        self.X_valid = X_filtered_valid
        self.y_valid = y_filtered_valid
        self.X_test = X_filtered_test
        self.y_test = y_filtered_test
        
        print("Data processing complete.")
        return (X_filtered_train, y_filtered_train, 
                X_filtered_valid, y_filtered_valid, 
                X_filtered_test, y_filtered_test)
    
    def _filter_and_map_data(self, features, labels, balance=False, max_count=None):
        """
        Filter data to include only specified classes and map to new labels
        
        Args:
            features (numpy.ndarray): Image features
            labels (numpy.ndarray): Labels
            balance (bool): Whether to balance classes
            max_count (int, optional): Maximum samples per class when balancing
            
        Returns:
            tuple: Filtered features and mapped labels
        """
        valid_classes = self.config['valid_classes']
        class_mapping = self.config['class_mapping']
        
        filtered_features = []
        filtered_labels = []
        
        if balance and max_count is not None:
            count_per_class = [0] * len(valid_classes)
            
            # First count samples per class
            for idx in range(len(labels)):
                if labels[idx] in valid_classes:
                    mapped_label = class_mapping[labels[idx]]
                    if count_per_class[mapped_label] < max_count:
                        filtered_features.append(features[idx])
                        filtered_labels.append(mapped_label)
                        count_per_class[mapped_label] += 1
            
            print(f"Balanced class distribution: {count_per_class}")
        else:
            # Not balancing, just filter
            for idx in range(len(labels)):
                if labels[idx] in valid_classes:
                    filtered_features.append(features[idx])
                    filtered_labels.append(class_mapping[labels[idx]])
        
        return np.array(filtered_features), np.array(filtered_labels)
    
    def create_datasets(self, augment_train=True, include_original=False):
        """
        Create PyTorch dataset objects
        
        Args:
            augment_train (bool): Whether to apply data augmentation to training data
            include_original (bool): Whether to include original images in dataset (for visualization comparison)
            
        Returns:
            tuple: Training, validation and test dataset objects
        """
        # Check if data is loaded
        if self.X_train is None or self.y_train is None:
            print("Error: Data not yet loaded. Please call load_data() method first.")
            return None, None, None
        
        print("Creating datasets...")
        img_size = self.config['img_size']
        mean = self.config['mean']
        std = self.config['std']
        
        # Define data transformations
        if augment_train:
            # Apply data augmentation to training data
            # TODO make some data augmentation
            train_transform = transforms.Compose([
                # Basic transformations
                transforms.Resize((img_size, img_size)),
                
                # Data augmentation
                transforms.ColorJitter(brightness=0.2, contrast=0.2), 
                transforms.RandomRotation(degrees=15),
                
                # Convert to tensor and normalize
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            print("Training data augmentation enabled")
        else:
            # No augmentation
            train_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
            print("Training data augmentation disabled")
        
        # For validation and test data, only basic processing
        eval_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # If including original images, define original image transformation
        orig_transform = None
        if include_original:
            orig_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
        
        # Create datasets
        train_dataset = TrafficSignDataset(
            self.X_train, self.y_train, 
            transform=train_transform, 
            orig_transform=orig_transform,
            class_mapping=self.class_names
        )
        
        valid_dataset = TrafficSignDataset(
            self.X_valid, self.y_valid, 
            transform=eval_transform, 
            orig_transform=orig_transform,
            class_mapping=self.class_names
        ) if self.X_valid is not None else None
        
        test_dataset = TrafficSignDataset(
            self.X_test, self.y_test, 
            transform=eval_transform, 
            orig_transform=orig_transform,
            class_mapping=self.class_names
        ) if self.X_test is not None else None
        
        print(f"Datasets created: {len(train_dataset)} training samples")
        if valid_dataset:
            print(f"{len(valid_dataset)} validation samples")
        if test_dataset:
            print(f"{len(test_dataset)} test samples")
        
        return train_dataset, valid_dataset, test_dataset
    
    def create_data_loaders(self, train_dataset, valid_dataset=None, test_dataset=None, batch_size=32, num_workers=4):
        """
        Create PyTorch data loaders
        
        Args:
            train_dataset, valid_dataset, test_dataset: Dataset objects
            batch_size (int): Batch size
            num_workers (int): Number of worker processes for data loading
            
        Returns:
            tuple: Training, validation and test data loaders
        """
        print(f"Creating data loaders (batch_size={batch_size}, workers={num_workers})...")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            drop_last=False,
            persistent_workers=num_workers > 0,
            prefetch_factor=num_workers > 0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        valid_loader = None
        if valid_dataset:
            valid_loader = DataLoader(
                valid_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                drop_last=False,
                persistent_workers=num_workers > 0,
                prefetch_factor=num_workers > 0,
                pin_memory=True if torch.cuda.is_available() else False
            )
        
        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                drop_last=False,
                persistent_workers=num_workers > 0,
                prefetch_factor=num_workers > 0,
                pin_memory=True if torch.cuda.is_available() else False
            )
        
        print(f"Data loaders created: {len(train_loader)} training batches")
        if valid_loader:
            print(f"{len(valid_loader)} validation batches")
        if test_loader:
            print(f"{len(test_loader)} test batches")
        
        return train_loader, valid_loader, test_loader
    
    def save_processed_data(self, output_dir='./processed_data', filename_prefix='traffic_sign'):
        """
        Save preprocessed data
        
        Args:
            output_dir (str): Output directory
            filename_prefix (str): Filename prefix
        """
        # Check if data is loaded
        if self.X_train is None or self.y_train is None:
            print("Error: Data not yet loaded. Please call load_data() method first.")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        train_file = os.path.join(output_dir, f"{filename_prefix}_train.npz")
        np.savez(train_file, features=self.X_train, labels=self.y_train)
        
        if self.X_valid is not None and self.y_valid is not None:
            valid_file = os.path.join(output_dir, f"{filename_prefix}_valid.npz")
            np.savez(valid_file, features=self.X_valid, labels=self.y_valid)
        
        if self.X_test is not None and self.y_test is not None:
            test_file = os.path.join(output_dir, f"{filename_prefix}_test.npz")
            np.savez(test_file, features=self.X_test, labels=self.y_test)
        
        # Save configuration
        config_file = os.path.join(output_dir, f"{filename_prefix}_config.json")
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        print(f"Preprocessed data saved to {output_dir}")
    
    def load_processed_data(self, data_dir, filename_prefix='traffic_sign'):
        """
        Load preprocessed data
        
        Args:
            data_dir (str): Data directory
            filename_prefix (str): Filename prefix
            
        Returns:
            tuple: Preprocessed training, validation and test data and labels
        """
        train_file = os.path.join(data_dir, f"{filename_prefix}_train.npz")
        valid_file = os.path.join(data_dir, f"{filename_prefix}_valid.npz")
        test_file = os.path.join(data_dir, f"{filename_prefix}_test.npz")
        config_file = os.path.join(data_dir, f"{filename_prefix}_config.json")
        
        # Load training data
        try:
            train_data = np.load(train_file)
            self.X_train = train_data['features']
            self.y_train = train_data['labels']
            
            # Load validation data (if exists)
            if os.path.exists(valid_file):
                valid_data = np.load(valid_file)
                self.X_valid = valid_data['features']
                self.y_valid = valid_data['labels']
            
            # Load test data (if exists)
            if os.path.exists(test_file):
                test_data = np.load(test_file)
                self.X_test = test_data['features']
                self.y_test = test_data['labels']
            
            # Load configuration (if exists)
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
            
            print(f"Preprocessed data loaded: {len(self.X_train)} training samples")
            if self.X_valid is not None:
                print(f"{len(self.X_valid)} validation samples")
            if self.X_test is not None:
                print(f"{len(self.X_test)} test samples")
            
            return (self.X_train, self.y_train, 
                    self.X_valid, self.y_valid, 
                    self.X_test, self.y_test)
        
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            raise