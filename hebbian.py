from tinygrad.tensor import Tensor
import math

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
               target_sparsity: float = 0.05):    # Target for inhibition sparsity
    # Initialize weights with Kaiming/He initialization
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound, requires_grad=False)
    # EMA templates for each neuron's preferred input pattern
    self.input_templates = Tensor.zeros(out_features, in_features)
    # EMA of each neuron's activation history (for homeostasis)
    self.activation_history = Tensor.zeros(out_features)
    # Parameters
    self.target_sparsity = target_sparsity#for inhibition
    self.template_alpha = template_alpha  # For learning templates
    self.activity_alpha = activity_alpha  # For homeostatic plasticity
    self.weight_norm = weight_norm
    # Storage for current state
    self.current_input = None
    self.current_output = None
    self.active_neurons = None  # Will store indices of neurons that survived inhibition


  def __call__(self, x: Tensor, label: int = None) -> Tensor:
    # Forward pass with detached weights to prevent graph building
    raw_act = x.linear(self.weight.T.detach()).realize()
    
    if label is not None:  # Training mode
        # Create winner mask and immediately realize it
        winner_mask = (Tensor.arange(self.weight.shape[0]) == label).float().reshape(-1, 1).realize()
        
        # Normalize input and realize
        input_norm = (x / (x.sum() + 1e-8)).realize()
        
        # Update templates - force evaluation at each step
        template_update = (1 - winner_mask * self.template_alpha)
        new_templates = (self.input_templates * template_update + 
                        x * winner_mask * self.template_alpha).realize()
        self.input_templates = new_templates
        del template_update, new_templates
        
        # Update activation history - force evaluation
        history_update = (1 - self.activity_alpha)
        new_history = (self.activation_history * history_update + 
                      winner_mask.reshape(-1) * self.activity_alpha).realize()
        self.activation_history = new_history
        del history_update, new_history
        
        # Update weights with normalization - force evaluation at each step
        weight_update = winner_mask * (input_norm - self.input_templates)
        new_weights = (self.weight + weight_update * raw_act.reshape(-1, 1)).realize()
        weight_norms = new_weights.sum(axis=1, keepdim=True).realize()
        final_weights = (new_weights * (self.weight_norm / (weight_norms + 1e-8))).realize()
        
        # Clean up intermediate tensors
        del weight_update, new_weights, weight_norms
        self.weight = final_weights
        del final_weights
        
        # Clean up input tensors
        del winner_mask, input_norm
    
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