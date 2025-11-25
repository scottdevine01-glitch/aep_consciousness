# AEP Framework Usage Guide

## Quick Start

### Neural Compression Analysis
```python
from modules.neural_compression import NeuralCompressionAnalyzer
import numpy as np

# Generate or load your neural data
# Shape: (n_timepoints, n_neurons)
data = np.random.randn(1000, 50)  

# Analyze compression metrics
analyzer = NeuralCompressionAnalyzer()
metrics = analyzer.compute_all_metrics(data)

print("Consciousness signature:", metrics)
