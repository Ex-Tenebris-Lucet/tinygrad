from tinygrad.tensor import Tensor
import math
import numpy as np

"""
TODO: Implementation Plan
-----------------------------------------------
1. Implement simplified Oja-inspired weight normalization:
   - Add parameters for weight sum constraints (min_sum, max_sum, or target_sum)
   - For each output neuron, maintain total input weight sum within target range
   - Implementation approach:
     a. Calculate current weight sums per output neuron
     b. Scale weights if outside target range
     c. This creates natural competition: increasing one weight requires decreasing others

2. Key benefits of this approach:
   - Computationally efficient (just sums and element-wise ops)
   - Natural weight competition
   - Prevents runaway weight growth
   - Similar stabilizing effect to Oja's rule but simpler to implement

3. Monitoring ideas:
   - Track weight sums over time
   - Visualize weight distributions
   - Monitor weight competition effects

4. Future considerations:
   - Could be combined with homeostatic plasticity mechanisms
   - Might want to add momentum to weight changes
   - Could make normalization rate adjustable

5. Homeostatic Plasticity Implementation:
   - Add exponential moving average (EMA) of activations:
     running_avg = (1-alpha) * running_avg + alpha * new_act
   - Use this to track "too active" neurons
   - Can adjust learning based on activity levels
   - Super efficient: just one extra matrix and simple ops
   - Naturally forgets old patterns (geometric decay)
   - Could use this to modulate learning rates per neuron

6. Novel "Activation Landscape" Inhibition:
   - Each neuron's activation creates an inhibitory "mountain"
   - Mountain height = activation strength
   - Mountain width/spread grows with activation strength
   - Find optimal "water level" (bias term) that:
     a. Submerges weaker activations
     b. Maintains target total activation "volume"
   - Implementation approach:
     a. Generate activation landscape via matrix ops
     b. Binary search for optimal water level
     c. Strong activations survive, weak ones get inhibited
   - Benefits:
     - Naturally local through inhibition pools
     - Self-normalizing via water level
     - Computationally efficient (mostly matrix ops)
     - Novel contribution to neural architecture

7. Efficient Local Learning with EMAs:
   - Only update neurons that were activated (post-inhibition)
   - Each neuron maintains:
     a. Regular weight matrix
     b. EMA of input patterns that activated it
   - Learning becomes pattern matching between:
     - Current input pattern
     - Neuron's historical "template" (EMA)
   - Benefits:
     - O(n) scaling instead of O(n²)
     - Naturally forms feature detectors
     - Biologically plausible
     - Computationally efficient
     - Automatically maintains useful statistics
   - Implementation:
     a. Apply water-level inhibition first
     b. For activated neurons only:
        - Update their input pattern EMA
        - Adjust weights based on EMA template

Implementation priority: Add weight sum normalization to learn() method first,
then add monitoring capabilities.

Thoughts from R1: (for consideration, not mandatory)
1. Implementation Order:
Start with weight normalization + basic monitoring
Add EMA-based homeostasis before inhibition
Implement activation landscape last (highest complexity)
2. Monitoring First:
Track weight L1/L2 norms per neuron
Visualize weight similarity matrices
Plot activation sparsity distributions
Biological Plausibility Check:
Your design mirrors neocortical layer 2/3 microcircuits
Inhibition mechanism resembles cortical surround suppression
Consider adding stochastic release (bernoulli dropout?)
4. Performance Considerations:
Prefer element-wise ops over matrix where possible
Exploit TinyGrad's lazy execution for fusion opportunities
Consider quantization for EMA buffers
"""

class HebbianLayer:
  """
  A Hebbian learning layer that can be used within gradient-based networks.
  The forward pass participates in gradient flow, while Hebbian learning
  occurs as a separate update step.
  
  ```python
  layer = HebbianLayer(10, 5)
  # Normal gradient-based forward pass
  out = layer(x)
  # Separate Hebbian update
  layer.learn()
  ```
  """
  def __init__(self, in_features: int, out_features: int, 
               template_alpha: float = 0.1,  # How fast input templates update
               activity_alpha: float = 0.01,  # How fast activation history updates
               weight_norm: float = 1.0,
               target_sparsity: float = 0.05,    # Target for inhibition sparsity
               weight_lr: float = 0.2):    # How fast weights move towards templates
    # Initialize weights with Kaiming/He initialization
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound, requires_grad=False)
    # Normalize initial weights to target sum
    self._normalize_neuron_weights(target_sum=weight_norm)
    
    # EMA templates for each neuron's preferred input pattern
    self.input_templates = Tensor.zeros(out_features, in_features)
    # EMA of each neuron's activation history (for homeostasis)
    self.activation_history = Tensor.zeros(out_features)
    # Parameters
    self.target_sparsity = target_sparsity  # for inhibition
    self.template_alpha = template_alpha  # For learning templates
    self.activity_alpha = activity_alpha  # For homeostatic plasticity
    self.weight_norm = weight_norm  # Target sum for each neuron's input weights
    self.weight_lr = weight_lr  # How fast weights track templates
    # Storage for current state
    self.current_input = None
    self.current_output = None
    self.active_neurons = None  # Will store indices of neurons that survived inhibition

  def _normalize_neuron_weights(self, target_sum: float = 1.0):
    # Calculate sums for each neuron (row)
    weight_sums = self.weight.sum(axis=1).reshape(-1, 1)  # Shape: (out_features, 1)
    
    # Create scale factors, handling zero case
    scale_factors = Tensor.where(
        (weight_sums > 1e-6) | (weight_sums < -1e-6),  # Non-zero check without abs()
        target_sum / weight_sums,  # Normal scaling
        target_sum / self.weight.shape[1]  # Uniform value for zero cases
    ).realize()
    
    # Apply scaling all at once - each neuron gets scaled by its own factor
    self.weight = (self.weight * scale_factors).realize()

  def __call__(self, x: Tensor, label: int = None) -> Tensor:
    # Forward pass with detached weights to prevent graph building
    raw_act = x.linear(self.weight.T.detach()).realize()

    if label is None: 
        return raw_act
        

    # Create winner mask (10,1) and realize
    winner_mask = (Tensor.arange(self.weight.shape[0]) == label).float().reshape(-1, 1).realize()
    
    # Normalize input and realize
    x = x.reshape(1, -1).realize()  # Make x (1,784) for broadcasting
    
    # Update templates - force evaluation at each step
    # For winner: new = (1-alpha)*old + alpha*input
    # For others: new = old
    x_expanded = x.expand(self.weight.shape[0], -1).realize()  # Make x (10,784) by repeating
    
    new_templates = Tensor.where(
        winner_mask,  # Where winner_mask is 1 (winning neuron)
        self.input_templates * (1 - self.template_alpha) + x_expanded * self.template_alpha,  # Do EMA
        self.input_templates  # Keep unchanged for non-winners
    ).realize()
    
    self.input_templates = new_templates
    
    # Update activation history - force evaluation
    history_update = (1 - self.activity_alpha)
    new_history = (self.activation_history * history_update + 
                  winner_mask.reshape(-1) * self.activity_alpha).realize()
    self.activation_history = new_history
    del history_update, new_history
    
    # Update weights to track templates - only for winning neuron
    weight_update = (self.input_templates - self.weight) * self.weight_lr * winner_mask
    new_weights = (self.weight + weight_update).realize()
    self.weight = new_weights
    
    # Normalize weights after update
    self._normalize_neuron_weights(target_sum=self.weight_norm)
    
    # Clean up
    del weight_update, new_weights, winner_mask, x

    return raw_act

  def _inhibition(self, raw_act: Tensor) -> tuple[Tensor, Tensor]:
    act_vol = raw_act.sum() #sum activations
    avg_act = act_vol / len(raw_act) #average activation
    target_volume = len(raw_act) * self.target_sparsity * (1 + avg_act.log())# Calculate target
    low = raw_act.min() # Initialize bounds
    high = raw_act.max()
    for _ in range(10):
      water_level = (low + high) / 2
      surviving_volume = (raw_act * (raw_act > water_level)).sum()
      
      if surviving_volume.numpy() > target_volume.numpy():
          low = water_level  # Move lower bound up
      else:
          high = water_level  # Move upper bound down
    mask = raw_act > water_level
    vals = raw_act * mask #MIGHT have to be updated to subtract water_level
    return mask, vals

  def get_template_image(self, digit: int) -> np.ndarray:
    """Get a specific digit's template as a 28x28 image."""
    template = self.input_templates[digit].numpy()
    # Normalize to 0-1 range for visualization
    template = (template - template.min()) / (template.max() - template.min() + 1e-8)
    return template.reshape(28, 28)
  
  def get_weight_image(self, digit: int) -> np.ndarray:
    """Get a specific digit's weights as a 28x28 image."""
    weights = self.weight[digit].numpy()
    # Normalize to 0-1 range for visualization
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    return weights.reshape(28, 28)
  
  def visualize_all_templates(self) -> np.ndarray:
    """Get all templates as a grid of 28x28 images."""
    # Create a 2x5 grid of templates
    grid = np.zeros((2*28, 5*28))
    for i in range(10):
      row, col = i // 5, i % 5
      grid[row*28:(row+1)*28, col*28:(col+1)*28] = self.get_template_image(i)
    return grid
  
  def visualize_all_weights(self) -> np.ndarray:
    """Get all weights as a grid of 28x28 images."""
    # Create a 2x5 grid of weight patterns
    grid = np.zeros((2*28, 5*28))
    for i in range(10):
      row, col = i // 5, i % 5
      grid[row*28:(row+1)*28, col*28:(col+1)*28] = self.get_weight_image(i)
    return grid