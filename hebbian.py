from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
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
                 template_alpha: float = 0.001,  # How fast templates learn (faster than weights)
                 activity_alpha: float = 0.001,  # How fast activation history updates
                 HSP_coeff: float = 1.5,
                 target_sparsity: float = 0.05):    # Target for inhibition sparsity
        
        self.template_alpha = template_alpha  # How fast templates learn
        self.activity_alpha = activity_alpha
        self.target_sparsity = target_sparsity
        self.num_neurons = out_features
        self.HSP_coeff = HSP_coeff
        
        self.weight = Tensor.uniform(out_features, in_features, low=-1, high=1, requires_grad=False)
        self.pos_template = Tensor.zeros(out_features, in_features)
        self.neg_template = Tensor.zeros(out_features, in_features)
        self.act_hist = Tensor.zeros(out_features)
        
    
    def __call__(self, x: Tensor, label: int = None) -> Tensor:
        relu_act = x.maximum(0) # relu the input tensor
        act_max = relu_act.max() # find the highest activation
        norm_act = (relu_act / act_max) # scale the activation tensor to 0-1

        out_act = norm_act.linear(self.weight.T.detach()).maximum(0).realize() #calculate the output activation
        out_max = out_act.max().realize()
        if out_max.item() > 0: out_act = (out_act / out_max).realize()

        if label is None: 
            return out_act  # output activations if inferring
            
        # Activation hist update
        self.act_hist = (self.act_hist * (1 - self.activity_alpha) 
                         + out_act * (self.activity_alpha)).realize()

        # Learning pass
        winner_mask = (Tensor.arange(self.weight.shape[0]) == label).float().reshape(-1, 1).realize()
        norm_act = norm_act.reshape(1, -1).realize()  # Make x (1,784)
        
        self.pos_template = Tensor.where( # Templates only remember inputs when they are activated
            winner_mask,
            self.pos_template * (1 - self.template_alpha) + norm_act * self.template_alpha,
            self.pos_template
        ).realize()

        out_act_reshape = out_act.reshape(-1,1)
        self.neg_template = Tensor.where(
            winner_mask,
            self.neg_template,
            self.neg_template * (1 - self.template_alpha * out_act_reshape) 
                - (norm_act * self.template_alpha * out_act_reshape)
        ).realize()
        
        ratios = (3.0 - self.HSP_coeff * (self.act_hist / (self.act_hist.max() + 1e-6))).reshape(-1, 1).clip(0.3, 3.0).realize()

        self.weight = (self.pos_template * ratios + self.neg_template).realize()
        
        return out_act

    '''def _normalize_neuron_weights(self, target_sum: float = 1.0, max_spread: float = 4.0):
        # First check spread and squish if needed
        spread = self.weight.max(axis=1) - self.weight.min(axis=1)  # Get spread for each neuron
        squish_factor = (Tensor([max_spread]) / (spread + 1e-6)).minimum(1.0).reshape(-1, 1)
        
        # If spread too large, squish weights toward their mean
        mean = self.weight.mean(axis=1, keepdim=True)
        squished = mean + (self.weight - mean) * squish_factor
        
        # Then normalize sum (on squished weights)
        weight_sums = squished.sum(axis=1).reshape(-1, 1)
        difference = target_sum - weight_sums
        adjustment = difference / self.weight.shape[1]
        
        # Apply both corrections
        self.weight = (squished + adjustment).realize()

    def _get_counter_learning_mask(self, out_act: Tensor, winner_mask: Tensor) -> Tensor:
        # Reshape everything for consistent broadcasting
        out_norm = (out_act / (out_act.max() + 1e-6)).reshape(-1, 1)
        
        # Only apply counter-learning to strongly activated non-winners
        strong_mask = (out_norm > 0.3).float() * (1 - winner_mask)
        
        # Only counter-learn if the neuron has developed a meaningful pattern
        pattern_strength = ((self.weight.max(axis=1) - self.weight.min(axis=1)) 
        / (self.weight_norm/2)).minimum(1.0).reshape(-1, 1)
        
        return strong_mask * pattern_strength'''

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
        template = self.pos_template[digit].numpy()
        # Scale by the reciprocal of the largest magnitude value
        max_magnitude = max(abs(template.max()), abs(template.min()))
        if max_magnitude != 0:  # Only scale if we have non-zero values
            template = template * (1.0 / max_magnitude)
        return template.reshape(28, 28)
    
    def get_neg_template_image(self, digit: int) -> np.ndarray:
        """Get a specific digit's negative template as a 28x28 image."""
        template = self.neg_template[digit].numpy()
        # Scale by the reciprocal of the largest magnitude value
        max_magnitude = max(abs(template.max()), abs(template.min()))
        if max_magnitude != 0:  # Only scale if we have non-zero values
            template = template * (1.0 / max_magnitude)
        return template.reshape(28, 28)
    
    def get_weight_image(self, digit: int) -> np.ndarray:
        """Get a specific digit's weights as a 28x28 image."""
        weights = self.weight[digit].numpy()
        # Scale by the reciprocal of the largest magnitude value
        max_magnitude = max(abs(weights.max()), abs(weights.min()))
        if max_magnitude != 0:  # Only scale if we have non-zero values
            weights = weights * (1.0 / max_magnitude)
        return weights.reshape(28, 28)

    def visualize_all_weights(self) -> np.ndarray:
        """Get all weights as a grid of 28x28 images."""
        # Create a 2x5 grid of weight patterns
        print(self.act_hist.numpy())
        grid = np.zeros((2*28, 5*28))
        for i in range(10):
            row, col = i // 5, i % 5
            grid[row*28:(row+1)*28, col*28:(col+1)*28] = self.get_weight_image(i)
        return grid

    def visualize_all_templates(self) -> np.ndarray:
        """Get all templates as a grid of 28x28 images.
        First 10 cells (2 rows) show positive templates,
        Next 10 cells (2 rows) show negative templates.
        """
        # Create a 4x5 grid of template patterns (pos and neg)
        grid = np.zeros((4*28, 5*28))
        
        # Fill the first 2 rows with positive templates
        for i in range(10):
            row, col = i // 5, i % 5
            grid[row*28:(row+1)*28, col*28:(col+1)*28] = self.get_template_image(i)
        
        # Fill the next 2 rows with negative templates
        for i in range(10):
            row, col = (i // 5) + 2, i % 5  # +2 to start at 3rd row
            grid[row*28:(row+1)*28, col*28:(col+1)*28] = self.get_neg_template_image(i)
            
        return grid