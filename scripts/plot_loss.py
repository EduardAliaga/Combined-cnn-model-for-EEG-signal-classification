import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def read_losses(file_path):
    """Read loss values from a file and return as a list of floats."""
    with open(file_path, 'r') as f:
        losses = [float(line.strip()) for line in f]
    return losses

def plot_losses(train_losses, val_losses, title, save_path):
    """Plot training and validation losses and save the plot."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)  # Save the plot to the specified path
    plt.close()  # Close the plot to free memory

# Base directory where the results are stored
base_directory = Path("/Users/aliag/Documents/4t/I2R/bci-disc-models/results")

# List of models to process
models = ["transformer"]

# Process each model
for model in models:
    # Find all directories that match the pattern for this model
    directories = [d for d in base_directory.glob(f"{model}*") if d.is_dir()]

    # Initialize lists to hold average losses
    avg_train_losses = []
    avg_val_losses = []

    # Process each directory for the current model
    for directory in directories:
        train_loss_file = directory / "train_epoch_losses.txt"
        val_loss_file = directory / "val_epoch_losses.txt"

        # Read the losses
        train_losses = read_losses(train_loss_file)
        val_losses = read_losses(val_loss_file)

        # Plot individual losses and save the plot in the same directory
        plot_title = f"Loss vs Epochs for {directory.name}"
        plot_losses(train_losses, val_losses, plot_title, directory / f"{plot_title}.png")

        # Add to average calculations
        if len(avg_train_losses) == 0:
            avg_train_losses = np.array(train_losses)
            avg_val_losses = np.array(val_losses)
        else:
            avg_train_losses += np.array(train_losses)
            avg_val_losses += np.array(val_losses)


    # Calculate and plot the average losses if directories were found
    if directories and len(avg_train_losses) > 0 and len(avg_val_losses) > 0:
        avg_train_losses /= len(directories)
        avg_val_losses /= len(directories)
        # Plot average losses and save the plot in the base directory
        avg_plot_title = f"Average Loss vs Epochs for {model}"
        plot_losses(avg_train_losses, avg_val_losses, avg_plot_title, base_directory / f"{avg_plot_title}.png")

