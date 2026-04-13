#!/usr/bin/env python
# coding: utf-8
"""
Extract specific prediction examples for the report.
Finds correct and incorrect predictions matching specific criteria and saves them as figures.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Import from existing codebase
from network import ResNet18
from dataset import TrafficSignProcessor

# Configuration
CHECKPOINT_PATH = 'checkpoint/best_model.pth'
RESULTS_DIR = 'results'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Class names mapping
CLASS_NAMES = {
    0: 'Stop',
    1: 'Turn right',
    2: 'Turn left',
    3: 'Ahead only',
    4: 'Roundabout mandatory'
}

# ImageNet normalization values (same as used in training)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denormalize_image(img_tensor):
    """Convert normalized tensor back to displayable image."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    img = STD * img + MEAN
    img = np.clip(img, 0, 1)
    return img


def calculate_brightness(img):
    """Calculate mean pixel intensity (brightness) of an image."""
    if isinstance(img, torch.Tensor):
        img = denormalize_image(img)
    # Convert to grayscale using luminance formula
    gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    return np.mean(gray)


def resize_for_display(img, size=128):
    """Resize image for better visibility in report."""
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    img_pil = img_pil.resize((size, size), Image.NEAREST)
    return np.array(img_pil) / 255.0


def save_single_image(img, title, color, save_path, figsize=(4, 4)):
    """Save a single image with title overlay."""
    fig, ax = plt.subplots(figsize=figsize)

    # Resize for visibility
    img_display = resize_for_display(img, size=128)

    ax.imshow(img_display)
    ax.set_title(title, color=color, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   - Saved to: {save_path}")


def run_inference(model, test_loader):
    """Run inference on entire test set and collect results."""
    model.eval()

    results = []  # List of dicts with all info needed

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confidences, predictions = probs.max(dim=1)

            for i in range(inputs.size(0)):
                img_tensor = inputs[i].cpu()
                img_denorm = denormalize_image(img_tensor)
                brightness = calculate_brightness(img_denorm)

                results.append({
                    'index': batch_idx * test_loader.batch_size + i,
                    'image': img_denorm,
                    'true_label': targets[i].item(),
                    'pred_label': predictions[i].item(),
                    'confidence': confidences[i].item() * 100,
                    'brightness': brightness,
                    'correct': predictions[i].item() == targets[i].item()
                })

    return results


def find_examples(results):
    """Find specific examples matching the criteria."""
    examples = {}

    # === CORRECT PREDICTIONS ===

    # 1. Dark Stop Sign - correctly classified Stop with lowest brightness
    correct_stops = [r for r in results if r['correct'] and r['true_label'] == 0]
    if correct_stops:
        darkest_stop = min(correct_stops, key=lambda x: x['brightness'])
        examples['dark_stop'] = darkest_stop

    # 2. Dark Turn Right - correctly classified Turn Right with lowest brightness
    correct_turn_right = [r for r in results if r['correct'] and r['true_label'] == 1]
    if correct_turn_right:
        darkest_turn_right = min(correct_turn_right, key=lambda x: x['brightness'])
        examples['dark_turn_right'] = darkest_turn_right

    # 3. High Confidence Turn Left - correctly classified Turn Left with highest confidence
    correct_turn_left = [r for r in results if r['correct'] and r['true_label'] == 2]
    if correct_turn_left:
        most_confident_turn_left = max(correct_turn_left, key=lambda x: x['confidence'])
        examples['confident_turn_left'] = most_confident_turn_left

    # === INCORRECT PREDICTIONS ===

    # 4. Turn Right misclassified as Roundabout
    turn_right_as_roundabout = [r for r in results if not r['correct']
                                 and r['true_label'] == 1 and r['pred_label'] == 4]
    if turn_right_as_roundabout:
        most_confident_wrong = max(turn_right_as_roundabout, key=lambda x: x['confidence'])
        examples['turnright_as_roundabout'] = most_confident_wrong

    # 5. Ahead Only misclassified as Roundabout
    ahead_as_roundabout = [r for r in results if not r['correct']
                           and r['true_label'] == 3 and r['pred_label'] == 4]
    if ahead_as_roundabout:
        # Pick medium-to-high confidence (sort and pick from upper half)
        sorted_cases = sorted(ahead_as_roundabout, key=lambda x: x['confidence'], reverse=True)
        # Pick one from upper quartile
        idx = len(sorted_cases) // 4
        examples['ahead_as_roundabout'] = sorted_cases[idx]

    # 6. Dark Failure - any incorrect prediction with lowest brightness
    incorrect = [r for r in results if not r['correct']]
    if incorrect:
        darkest_failure = min(incorrect, key=lambda x: x['brightness'])
        examples['dark_failure'] = darkest_failure

    return examples


def print_and_save_examples(examples):
    """Print summary and save all example images."""

    print("\n" + "="*60)
    print("=== CORRECT PREDICTIONS ===")
    print("="*60)

    # 1. Dark Stop Sign
    if 'dark_stop' in examples:
        ex = examples['dark_stop']
        print(f"\n1. Dark Stop Sign")
        print(f"   - Test set index: {ex['index']}")
        print(f"   - True class: {CLASS_NAMES[ex['true_label']]}")
        print(f"   - Predicted: {CLASS_NAMES[ex['pred_label']]}")
        print(f"   - Confidence: {ex['confidence']:.1f}%")
        print(f"   - Mean pixel intensity: {ex['brightness']:.3f}")

        title = f"Predicted: {CLASS_NAMES[ex['pred_label']]} ({ex['confidence']:.1f}%)"
        save_single_image(ex['image'], title, 'green',
                         os.path.join(RESULTS_DIR, 'report_correct_stop_dark.png'))

    # 2. Dark Turn Right
    if 'dark_turn_right' in examples:
        ex = examples['dark_turn_right']
        print(f"\n2. Dark Turn Right")
        print(f"   - Test set index: {ex['index']}")
        print(f"   - True class: {CLASS_NAMES[ex['true_label']]}")
        print(f"   - Predicted: {CLASS_NAMES[ex['pred_label']]}")
        print(f"   - Confidence: {ex['confidence']:.1f}%")
        print(f"   - Mean pixel intensity: {ex['brightness']:.3f}")

        title = f"Predicted: {CLASS_NAMES[ex['pred_label']]} ({ex['confidence']:.1f}%)"
        save_single_image(ex['image'], title, 'green',
                         os.path.join(RESULTS_DIR, 'report_correct_turnright_dark.png'))

    # 3. High Confidence Turn Left
    if 'confident_turn_left' in examples:
        ex = examples['confident_turn_left']
        print(f"\n3. High Confidence Turn Left")
        print(f"   - Test set index: {ex['index']}")
        print(f"   - True class: {CLASS_NAMES[ex['true_label']]}")
        print(f"   - Predicted: {CLASS_NAMES[ex['pred_label']]}")
        print(f"   - Confidence: {ex['confidence']:.1f}%")
        print(f"   - Mean pixel intensity: {ex['brightness']:.3f}")

        title = f"Predicted: {CLASS_NAMES[ex['pred_label']]} ({ex['confidence']:.1f}%)"
        save_single_image(ex['image'], title, 'green',
                         os.path.join(RESULTS_DIR, 'report_correct_turnleft_confident.png'))

    print("\n" + "="*60)
    print("=== INCORRECT PREDICTIONS ===")
    print("="*60)

    # 4. Turn Right -> Roundabout
    if 'turnright_as_roundabout' in examples:
        ex = examples['turnright_as_roundabout']
        print(f"\n4. Turn Right -> Roundabout")
        print(f"   - Test set index: {ex['index']}")
        print(f"   - True class: {CLASS_NAMES[ex['true_label']]}")
        print(f"   - Predicted: {CLASS_NAMES[ex['pred_label']]}")
        print(f"   - Confidence: {ex['confidence']:.1f}%")
        print(f"   - Mean pixel intensity: {ex['brightness']:.3f}")

        title = f"Predicted: {CLASS_NAMES[ex['pred_label']]} ({ex['confidence']:.1f}%)\nTrue: {CLASS_NAMES[ex['true_label']]}"
        save_single_image(ex['image'], title, 'red',
                         os.path.join(RESULTS_DIR, 'report_incorrect_turnright_as_roundabout.png'))
    else:
        print("\n4. Turn Right -> Roundabout: NOT FOUND (no such misclassification in test set)")

    # 5. Ahead Only -> Roundabout
    if 'ahead_as_roundabout' in examples:
        ex = examples['ahead_as_roundabout']
        print(f"\n5. Ahead Only -> Roundabout")
        print(f"   - Test set index: {ex['index']}")
        print(f"   - True class: {CLASS_NAMES[ex['true_label']]}")
        print(f"   - Predicted: {CLASS_NAMES[ex['pred_label']]}")
        print(f"   - Confidence: {ex['confidence']:.1f}%")
        print(f"   - Mean pixel intensity: {ex['brightness']:.3f}")

        title = f"Predicted: {CLASS_NAMES[ex['pred_label']]} ({ex['confidence']:.1f}%)\nTrue: {CLASS_NAMES[ex['true_label']]}"
        save_single_image(ex['image'], title, 'red',
                         os.path.join(RESULTS_DIR, 'report_incorrect_aheadonly_as_roundabout.png'))

    # 6. Dark Failure
    if 'dark_failure' in examples:
        ex = examples['dark_failure']
        print(f"\n6. Dark Failure")
        print(f"   - Test set index: {ex['index']}")
        print(f"   - True class: {CLASS_NAMES[ex['true_label']]}")
        print(f"   - Predicted: {CLASS_NAMES[ex['pred_label']]}")
        print(f"   - Confidence: {ex['confidence']:.1f}%")
        print(f"   - Mean pixel intensity: {ex['brightness']:.3f}")

        title = f"Predicted: {CLASS_NAMES[ex['pred_label']]} ({ex['confidence']:.1f}%)\nTrue: {CLASS_NAMES[ex['true_label']]}"
        save_single_image(ex['image'], title, 'red',
                         os.path.join(RESULTS_DIR, 'report_incorrect_dark_failure.png'))

    return examples


def create_combined_grid(examples):
    """Create a 2x3 grid with all 6 examples."""

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Top row: correct predictions
    correct_keys = ['dark_stop', 'dark_turn_right', 'confident_turn_left']
    correct_titles = ['Dark Stop Sign', 'Dark Turn Right', 'High Confidence Turn Left']

    for i, (key, subtitle) in enumerate(zip(correct_keys, correct_titles)):
        ax = axes[0, i]
        if key in examples:
            ex = examples[key]
            img_display = resize_for_display(ex['image'], size=128)
            ax.imshow(img_display)
            title = f"{subtitle}\nPred: {CLASS_NAMES[ex['pred_label']]} ({ex['confidence']:.1f}%)"
            ax.set_title(title, color='green', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Not Found', ha='center', va='center')
            ax.set_title(subtitle, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom row: incorrect predictions
    incorrect_keys = ['turnright_as_roundabout', 'ahead_as_roundabout', 'dark_failure']
    incorrect_titles = ['Turn Right -> Roundabout', 'Ahead Only -> Roundabout', 'Dark Failure']

    for i, (key, subtitle) in enumerate(zip(incorrect_keys, incorrect_titles)):
        ax = axes[1, i]
        if key in examples:
            ex = examples[key]
            img_display = resize_for_display(ex['image'], size=128)
            ax.imshow(img_display)
            title = f"{subtitle}\nPred: {CLASS_NAMES[ex['pred_label']]} | True: {CLASS_NAMES[ex['true_label']]}"
            ax.set_title(title, color='red', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Not Found', ha='center', va='center')
            ax.set_title(subtitle, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Add row labels
    axes[0, 0].set_ylabel("Correct", fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel("Incorrect", fontsize=14, fontweight='bold')

    fig.suptitle("Representative Correct and Incorrect Predictions", fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, 'report_success_failure_grid.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n=== Combined grid saved to: {save_path} ===")


def main():
    print("="*60)
    print("Extracting Report Images for Task 2")
    print("="*60)

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load model
    print(f"\nLoading model from {CHECKPOINT_PATH}...")
    model = ResNet18(num_classes=5)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded successfully. Using device: {DEVICE}")

    # Load test data
    print("\nLoading test dataset...")
    processor = TrafficSignProcessor()
    processor.load_data('train.p', 'valid.p', 'test.p')

    # Create datasets with NO augmentation for test
    _, _, test_dataset = processor.create_datasets(augment_train=False, include_original=False)

    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )
    print(f"Test set loaded: {len(test_dataset)} samples")

    # Run inference on entire test set
    print("\nRunning inference on test set...")
    results = run_inference(model, test_loader)
    print(f"Inference complete. Processed {len(results)} samples.")

    # Count correct/incorrect
    correct_count = sum(1 for r in results if r['correct'])
    print(f"Correct: {correct_count}/{len(results)} ({100*correct_count/len(results):.2f}%)")

    # Find specific examples
    print("\nSearching for specific examples...")
    examples = find_examples(results)

    # Print summary and save individual images
    examples = print_and_save_examples(examples)

    # Create combined grid
    print("\nCreating combined grid figure...")
    create_combined_grid(examples)

    print("\n" + "="*60)
    print("DONE! All images saved to results/ directory")
    print("="*60)


if __name__ == '__main__':
    main()
