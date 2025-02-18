from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad import nn
from tinygrad.helpers import getenv, trange
from tinygrad.nn.datasets import mnist
from tinygrad.nn.hebbian import HebbianLayer
import numpy as np
from datetime import datetime
import os
import plotext as plt

# Full beautiful_mnist model
class Model:
  def __init__(self):
    self.layers = [
      nn.Conv2d(1, 32, 5),  # Full size conv layers from beautiful_mnist
      Tensor.relu,
      nn.Conv2d(32, 32, 5),
      Tensor.relu,
      nn.BatchNorm(32),
      Tensor.max_pool2d,
      nn.Conv2d(32, 64, 3),
      Tensor.relu,
      nn.Conv2d(64, 64, 3),
      Tensor.relu,
      nn.BatchNorm(64),
      Tensor.max_pool2d,
      lambda x: x.flatten(1),
      nn.Linear(576, 10)
    ]

  def __call__(self, x): 
    return x.sequential(self.layers)

if __name__ == "__main__":
  try:
    print("Loading MNIST data...")
    X_train, Y_train, X_test, Y_test = mnist()
    
    # Take exactly 1024 training samples
    X_train, Y_train = X_train[:1024], Y_train[:1024]
    # Take exactly 128 test samples
    X_test, Y_test = X_test[:128], Y_test[:128]
    print(f"Using {len(X_train)} training samples and {len(X_test)} test samples")

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Training parameters - hardcoded to match requirements
    BS = 32  # Exactly 32 samples per batch
    STEPS = 32  # Exactly 32 steps (32*32 = 1024 total samples)
    
    # Save training parameters
    run_params = {
      'n_train': len(X_train),
      'n_test': len(X_test),
      'batch_size': BS,
      'steps': STEPS,
      'timestamp': timestamp
    }

    print("Creating model...")
    model = Model()
    opt = nn.optim.Adam(nn.state.get_parameters(model))
    print("Model created!")
    
    # Track accuracies for plotting learning curve
    accuracies = []
    losses = []

    # Setup live plotting
    plt.clf()
    plt.title("Training Progress")
    plt.xlabel("Step")
    plt.ylabel("Value")

    print(f"Training for {STEPS} steps with batch size {BS} (total {STEPS*BS} samples)...")
    for i in (t := trange(STEPS)):
      with Tensor.train():
        # Get random batch
        samples = Tensor.randint(BS, high=X_train.shape[0])
        
        # Training step
        opt.zero_grad()
        out = model(X_train[samples])
        loss = out.sparse_categorical_crossentropy(Y_train[samples])
        loss.backward()
        opt.step()

      # Test accuracy every step
      with Tensor.test():
        test_acc = (model(X_test).argmax(axis=1) == Y_test).mean().numpy() * 100
        losses.append(loss.numpy())
        accuracies.append(test_acc)
        t.set_description(f"Step {i+1}/{STEPS} - loss: {loss.numpy():6.2f} test_accuracy: {test_acc:5.2f}%")
      
      # Update plot every step
      plt.clf()
      # Plot both on same axes but with different markers
      plt.plot(accuracies, marker="dot", label="Accuracy")
      plt.plot(losses, marker="cross", label="Loss")
      plt.grid(True)
      plt.show()

    print("\nFinal Results:")
    print(f"Starting accuracy: {accuracies[0]:.2f}%")
    print(f"Final accuracy: {accuracies[-1]:.2f}%")
    print(f"Accuracy gain per step: {(accuracies[-1] - accuracies[0])/STEPS:.2f}%")
    
    # Save results
    results = {
      'accuracies': np.array(accuracies),
      'losses': np.array(losses),
      'params': run_params
    }
    
    save_path = f'results/backprop_run_{timestamp}.npz'
    np.savez(save_path, **results)
    print(f"\nResults saved to {save_path}")
    
    print("Done!")
    
  except Exception as e:
    print(f"\nError occurred: {str(e)}")
    raise  # Re-raise the exception for full traceback if needed 