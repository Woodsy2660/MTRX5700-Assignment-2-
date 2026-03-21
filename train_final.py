#!/usr/bin/env python
# coding: utf-8

"""
Traffic Sign Classification Training Pipeline
-------------------------------------------------
Main script for training the traffic sign classification model with ResNet18
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import argparse
from tqdm import tqdm
import numpy as np

# Import custom modules
from network import ResNet18
from dataset import TrafficSignProcessor, set_seed
import vis_utils


# Device selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Training function
def train(model, train_loader, optimizer, criterion, epoch, epochs):
    """
    Train model for one epoch
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer
        criterion: Loss criterion
        epoch (int): Current epoch
        epochs (int): Total number of epochs
        
    Returns:
        tuple: Average loss and accuracy for the epoch
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # Using tqdm progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        # If the dataset contains original images, ignore them
        if isinstance(inputs, list) and len(inputs) > 1:
            inputs = inputs[0]
        
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Statistics
        # TODO calculate the loss and acc

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()
    
    avg_loss = train_loss / len(train_loader)

    acc = 100. * correct / total
    
    return avg_loss, acc


# Validation function
def validate(model, val_loader, criterion):
    """
    Validate model on validation set
    
    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        criterion: Loss criterion
        
    Returns:
        tuple: Average loss and accuracy for validation
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # If the dataset contains original images, ignore them
            if isinstance(inputs, list) and len(inputs) > 1:
                inputs = inputs[0]
            
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Statistics
            # TODO calculate the loss and acc

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()
    
    avg_loss = val_loss / len(val_loader)

    acc = 100. * correct / total
    
    return avg_loss, acc


# Save checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, acc, history, filepath):
    """
    Save model checkpoint
    
    Args:
        model (nn.Module): Model to save
        optimizer (Optimizer): Optimizer state
        scheduler: Learning rate scheduler state
        epoch (int): Current epoch
        acc (float): Validation accuracy
        history (tuple): Training history
        filepath (str): Path to save the checkpoint
    """
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'acc': acc,
        'epoch': epoch,
        'train_history': history
    }
    
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    
    # Save an additional best_model.pth for easier loading
    best_filepath = os.path.dirname(filepath) + '/best_model.pth'
    torch.save(state, best_filepath)
    print(f"Best model saved to {best_filepath}")


# Main function
def main():
    """
    Main function for training the traffic sign classifier
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Traffic Sign ResNet18 Classification')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--optimizer', default='rmsprop', type=str, help='Optimizer (sgd, adam, rmsprop)')
    parser.add_argument('--scheduler', default='cosine', type=str, help='Learning rate scheduler (cosine, step, none)')
    parser.add_argument('--data_path', default=None, type=str, help='Data path')
    parser.add_argument('--use_processed', action='store_true', help='Use preprocessed data')
    parser.add_argument('--workers', default=1, type=int, help='Number of data loading threads')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--step_size', default=30, type=int, help='Learning rate step size')
    parser.add_argument('--gamma', default=0.1, type=float, help='Learning rate decay rate')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize training history and state variables
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_acc = 0
    start_epoch = 0  # Important: ensure start_epoch has a default value
    
    # Load data
    processor = TrafficSignProcessor()
    
    if args.use_processed and os.path.exists('./processed_data'):
        processor.load_processed_data('./processed_data')
    else:
        training_file = 'train.p' if args.data_path is None else os.path.join(args.data_path, 'train.p')
        validation_file = 'valid.p' if args.data_path is None else os.path.join(args.data_path, 'valid.p')
        testing_file = 'test.p' if args.data_path is None else os.path.join(args.data_path, 'test.p')
        
        processor.load_data(training_file, validation_file, testing_file)
    
    # Create datasets
    train_dataset, valid_dataset, test_dataset = processor.create_datasets(augment_train=True, include_original=False)
    train_loader, val_loader, test_loader = processor.create_data_loaders(
        train_dataset, valid_dataset, test_dataset, batch_size=args.batch_size, num_workers=args.workers)
    
    # Create directory to save checkpoints
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    
    # Initialize model
    model = ResNet18()
    model = model.to(device)
    
    # Data parallel (if multiple GPUs)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Initialize checkpoint variable
    checkpoint = None
    
    # If resuming from checkpoint
    if args.resume:
        # Try to load checkpoint
        print('Resuming from checkpoint..')
        if not os.path.isdir('checkpoint'):
            print('Error: No checkpoint directory found!')
        else:
            # Build filename
            opt_name = args.optimizer.lower()
            lr_str = f"_lr_{args.lr}" if args.lr != 0.01 else ""
            scheduler_str = "" if args.scheduler.lower() != "none" else "_no_scheduler"
            
            checkpoint_path = f'./checkpoint/ckpt_{opt_name}{lr_str}{scheduler_str}.pth'
            
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Specified checkpoint {checkpoint_path} not found, trying to load best_model.pth")
                checkpoint_path = './checkpoint/best_model.pth'
                
                if not os.path.exists(checkpoint_path):
                    print(f"Warning: best_model.pth not found, starting training from scratch")
                else:
                    # Load checkpoint
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint['model'])
                    best_acc = checkpoint['acc']
                    start_epoch = checkpoint['epoch']
                    
                    # Restore training history
                    if 'train_history' in checkpoint:
                        history = checkpoint['train_history']
                        train_loss_list, train_acc_list, val_loss_list, val_acc_list = history
                    
                    print(f"Resumed from checkpoint Epoch: {start_epoch}, Accuracy: {best_acc:.2f}%")
            else:
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
                best_acc = checkpoint['acc']
                start_epoch = checkpoint['epoch']
                
                # Restore training history
                if 'train_history' in checkpoint:
                    history = checkpoint['train_history']
                    train_loss_list, train_acc_list, val_loss_list, val_acc_list = history
                
                print(f"Resumed from checkpoint Epoch: {start_epoch}, Accuracy: {best_acc:.2f}%")
    
    # Loss function
    # TODO Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Set optimizer
    # TODO set the optimizer
    if args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # If resuming from checkpoint, also restore optimizer state
    if checkpoint is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Learning rate scheduler
    if args.scheduler.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None
    
    # If resuming from checkpoint, also restore scheduler state
    if checkpoint is not None and scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Early stopping variables
    no_improve_epochs = 0
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train one epoch
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, args.epochs)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
        
        # Check if this is the best model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        # Save history
        history = (train_loss_list, train_acc_list, val_loss_list, val_acc_list)
        
        # Build checkpoint filename based on optimizer and learning rate
        opt_name = args.optimizer.lower()
        lr_str = f"_lr_{args.lr}" if args.lr != 0.01 else ""
        scheduler_str = "" if args.scheduler.lower() != "none" else "_no_scheduler"
        filepath = f'./checkpoint/ckpt_{opt_name}{lr_str}{scheduler_str}.pth'
        
        # Save checkpoint
        if is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, history, filepath)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
    
    # Test the best model
    print("\nTesting the best model...")
    
    # Load the best model
    best_model_path = './checkpoint/best_model.pth'
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model'])
    else:
        print("Warning: Best model not found, using current model for testing")
    
    # Test
    test_results = vis_utils.plot_confusion_matrix(model, test_loader, processor.class_names, device, criterion)
    
    # Visualize results
    vis_utils.visualize_training_results(
        train_loss_list, train_acc_list, val_loss_list, val_acc_list, 
        best_acc, args.optimizer, args.lr, args.batch_size, args.scheduler
    )
    
    # Visualize predictions
    vis_utils.visualize_predictions(model, test_loader, processor.class_names, device)
    
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%, Test accuracy: {test_results['accuracy']:.2f}%")


if __name__ == '__main__':
    main()