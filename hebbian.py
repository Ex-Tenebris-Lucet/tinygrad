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
    def __init__(self, in_features: int, out_features: int, 
                 learning_alpha: float = 0.1,  # How fast weights update 
                 template_alpha: float = 0.1,  # How fast templates learn (faster than weights)
                 activity_alpha: float = 0.01,  # How fast activation history updates
                 weight_norm: float = 1.0,
                 target_sparsity: float = 0.05):    # Target for inhibition sparsity
        # Initialize weights with small positive values
        bound = 0.1 / math.sqrt(in_features)
        self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound, requires_grad=False)
        
        # Initialize templates to zeros (clean slate for learning)
        self.templates = Tensor.zeros(out_features, in_features)
        
        # Parameters
        self.template_alpha = template_alpha  # How fast templates learn
        self.learning_alpha = learning_alpha  # How fast weights track templates
        self.activity_alpha = activity_alpha
        self.weight_norm = weight_norm
        self.target_sparsity = target_sparsity
        
        # Storage for current state
        self.activation_history = Tensor.zeros(out_features)
        self.current_input = None
        self.current_output = None
        self.active_neurons = None
    
    def __call__(self, x: Tensor, label: int = None) -> Tensor:
        relu_act = x.maximum(0) #relu the input tensor
        act_max = relu_act.max() #find the highest activation
        norm_act = relu_act / act_max #scale the activation tensor to 0-1

        out_act = norm_act.linear(self.weight.T.detach()) #calculate the output activation

        if label is None: 
            return out_act.realize()  # Only realize at the end
            
        # Learning pass
        winner_mask = (Tensor.arange(self.weight.shape[0]) == label).float().reshape(-1, 1)
        
        counter_learning_mask = self._get_counter_learning_mask(out_act)
        
        # Update templates with both positive and counter-associative learning
        self.templates = Tensor.where(
            winner_mask,
            # Winner: normal positive learning
            self.templates * (1 - self.template_alpha) + norm_act * self.template_alpha,
            # Counter learning for strong and wrong neurons
            Tensor.where(
                out_act.reshape(-1, 1) < 0.5,  # if weakly activated
                self.templates,  # do nothing
                self.templates - (norm_act * 0.1)  # gentle counter-learning, no fancy scaling
            )
        ).realize()
        
        # Update weights to follow templates
        self.weight = Tensor.where(
            winner_mask,
            self.weight + (self.templates - self.weight) * self.learning_alpha,
            self.weight
        ).realize()
        
        return out_act.realize()


    def _normalize_templates(self, max_spread: float = 3.0):
        templates_abs = self.templates.abs()
        scale_factor = 1.0 / (1.0 + templates_abs/max_spread)  # Will be close to 1 for small values, smaller for large values
        self.templates = (self.templates * scale_factor).realize()

    def _get_counter_learning_mask(self, out_act: Tensor) -> Tensor:
        # Reshape everything to (10,1) for consistent broadcasting
        out_norm = (out_act / out_act.max()).reshape(-1, 1)
        strong_mask = (out_norm > 0.5).float()
        pattern_strength = ((self.weight.max(axis=1) - self.weight.min(axis=1)) / (self.weight_norm)).minimum(1.0).reshape(-1, 1)
        return strong_mask * pattern_strength

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
        template = self.templates[digit].numpy()
        
        # Diagnostic information about value distribution
        flat_template = template.flatten()
        unique, counts = np.unique(flat_template, return_counts=True)
        mode_value = unique[counts.argmax()]
        print(f"Digit {digit} stats - Mode: {mode_value:.6f}, Min: {template.min():.6f}, Max: {template.max():.6f}")
        
        # Scale by the reciprocal of the largest magnitude value
        max_magnitude = max(abs(template.max()), abs(template.min()))
        if max_magnitude != 0:  # Only scale if we have non-zero values
            template = template * (1.0 / max_magnitude)
        return template.reshape(28, 28)
    
    def get_weight_image(self, digit: int) -> np.ndarray:
        """Get a specific digit's weights as a 28x28 image."""
        weights = self.weight[digit].numpy()
        
        # Diagnostic information about value distribution
        flat_weights = weights.flatten()
        unique, counts = np.unique(flat_weights, return_counts=True)
        mode_value = unique[counts.argmax()]
        print(f"Weight {digit} stats - Mode: {mode_value:.6f}, Min: {weights.min():.6f}, Max: {weights.max():.6f}")
        
        # Scale by the reciprocal of the largest magnitude value
        max_magnitude = max(abs(weights.max()), abs(weights.min()))
        if max_magnitude != 0:  # Only scale if we have non-zero values
            weights = weights * (1.0 / max_magnitude)
        return weights.reshape(28, 28)

    def visualize_all_weights(self) -> np.ndarray:
        """Get all weights as a grid of 28x28 images."""
        # Create a 2x5 grid of weight patterns
        grid = np.zeros((2*28, 5*28))
        for i in range(10):
            row, col = i // 5, i % 5
            grid[row*28:(row+1)*28, col*28:(col+1)*28] = self.get_weight_image(i)
        return grid

    def visualize_all_templates(self) -> np.ndarray:
        """Get all templates as a grid of 28x28 images."""
        # Create a 2x5 grid of template patterns
        grid = np.zeros((2*28, 5*28))
        for i in range(10):
            row, col = i // 5, i % 5
            grid[row*28:(row+1)*28, col*28:(col+1)*28] = self.get_template_image(i)
        return grid