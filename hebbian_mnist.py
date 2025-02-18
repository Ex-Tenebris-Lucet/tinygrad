from tinygrad.tensor import Tensor
import numpy as np
from tinygrad.nn.datasets import mnist
from tinygrad.nn.hebbian import HebbianLayer
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import trange

# Simple classifier - one Hebbian layer
class SimpleHebbianClassifier:
    def __init__(self):
        self.hebb = HebbianLayer(784, 10)  # 784 -> 10 digits
    
    def __call__(self, x, label=None):
        # For single sample processing only
        x_flat = x.reshape(-1)  # -> (784,)
        return self.hebb(x_flat, label)

if __name__ == "__main__":
    # Load data - just take what we need
    X_train, Y_train, X_test, Y_test = mnist()
    X_train, Y_train = X_train[:1024], Y_train[:1024]  # 1024 training samples
    X_test, Y_test = X_test[:128], Y_test[:128]  # 128 test samples
    
    print(f"Training on {len(X_train)} samples")
    model = SimpleHebbianClassifier()
    
    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        # Get single random index
        idx = Tensor.randint(1, high=X_train.shape[0])
        return model(X_train[idx], Y_train[idx])
    
    @TinyJit
    @Tensor.test()
    def test_step(idx) -> Tensor:
        # Test a specific sample
        pred = model(X_test[idx])
        return (pred.argmax() == Y_test[idx]).realize()
    
    # Training loop
    for i in (t := trange(len(X_train))):
        # Train on single sample using random index
        train_step()
        
        # Test every 128 samples
        if (i + 1) % 128 == 0:
            # Test ALL samples in order
            correct = 0
            for j in range(len(X_test)):
                if test_step(Tensor([j])).numpy():
                    correct += 1
            test_acc = (correct / len(X_test)) * 100
            t.set_description(f"Step {i+1}: {test_acc:.1f}% accuracy")
