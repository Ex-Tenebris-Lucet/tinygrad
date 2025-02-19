import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.tinygrad import GlobalCounters
from tinygrad.nn.datasets import mnist
from tinygrad.nn.hebbian import HebbianLayer
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import trange
from collections import Counter  # For safe counting
import matplotlib.pyplot as plt

# Simple classifier - one Hebbian layer
class SimpleHebbianClassifier:
    def __init__(self):
        self.hebb = HebbianLayer(784, 10)  # 784 -> 10 digits
    
    def __call__(self, x, label=None):  # Put the label parameter back
        # For single sample processing only
        x_flat = x.reshape(-1)  # -> (784,)
        return self.hebb(x_flat, label)  # Pass label to HebbianLayer

if __name__ == "__main__":
    # Load MNIST - we'll use the test set as our training data since it's smaller
    X_train, Y_train, X_test, Y_test = mnist()
    X = X_test  # Use test set (it's smaller and that's fine for testing our idea)
    Y = Y_test
    print(f"\nTraining on {len(X)} samples")
    
    # Create our model
    model = SimpleHebbianClassifier()
    
    # Single step that handles both prediction and learning
    #@TinyJit
    def step(i: Tensor) -> tuple[Tensor, Tensor]:
        # Get sample using tensor indexing
        x = X[i]
        y = Y[i]
        # First predict
        with Tensor.test():
            out = model(x)
            pred = out.argmax()  # No axis needed for 1D tensor
        # Check if correct
        correct = (pred == y)
        # Learn
        model(x, y)
        return correct, pred
    
    # Training loop with live accuracy tracking
    print("\nTraining Progress:")
    correct_count = 0
    predictions = []
    
    # Safe prediction distribution tracking
    pred_counter = Counter()  # Counts predictions in current window
    window_size = 300  # How many predictions to track at once
    
    # Setup figure for visualization
    plt.ion()  # Interactive mode on
    
    for i in (t := trange(len(X))):
        GlobalCounters.reset()
        
        # Create index tensor for this step
        idx = Tensor([i])
        
        # Predict and learn
        was_correct, pred = step(idx)
        pred_int = int(pred.numpy())  # Safely convert to Python int
        predictions.append(pred_int)
        
        # Update prediction counter
        pred_counter[pred_int] += 1
        
        # Update accuracy stats
        if was_correct.numpy():
            correct_count += 1
        
        # Show live accuracy and recent distribution every 10 samples
        if (i + 1) % 10 == 0:
            acc = (correct_count / (i + 1)) * 100
            unique, counts = np.unique(predictions[-10:], return_counts=True)
            dist = dict(zip(map(int, unique), map(int, counts)))
            t.set_description(f"Live Accuracy: {acc:5.2f}% - Last 10 preds: {dist}")
        
        # Show full distribution and visualizations every 500 samples
        if (i + 1) % window_size == 0:
            print("\nPrediction distribution over last 500 samples:")
            for digit in range(10):  # For all possible digits
                print(f"{digit}: {pred_counter[digit]}")
            
            # Create figure with 3 subplots
            plt.figure(figsize=(15, 5))
            
            # Plot 1: Templates
            plt.subplot(131)
            plt.imshow(model.hebb.visualize_all_templates(), cmap='gray')
            plt.title("Templates")
            plt.axis('off')
            
            # Plot 2: Weights
            plt.subplot(132)
            plt.imshow(model.hebb.visualize_all_weights(), cmap='gray')
            plt.title("Weights")
            plt.axis('off')
            
            # Plot 3: Distribution
            plt.subplot(133)
            plt.bar(range(10), [pred_counter[i] for i in range(10)])
            plt.title("Prediction Distribution")
            plt.xlabel("Digit")
            plt.ylabel("Count")
            
            plt.tight_layout()
            plt.show()
            plt.pause(0.1)  # Small pause to ensure display
            
            print()  # Empty line for readability
            pred_counter.clear()  # Reset counter for next window
    
    plt.ioff()  # Turn off interactive mode
    