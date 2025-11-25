"""
Example: Using Neural Compression Analysis with Real Data Patterns
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.neural_compression import NeuralCompressionAnalyzer

def create_conscious_like_data():
    """Create data that mimics conscious state characteristics"""
    np.random.seed(42)
    n_time = 2000
    n_neurons = 40
    
    # Create highly structured, integrated data
    t = np.linspace(0, 20, n_time)
    
    # Multiple frequency components with phase relationships
    base_signals = [
        np.sin(2 * np.pi * 4 * t),
        np.sin(2 * np.pi * 8 * t + 0.5),  # Phase shifted
        np.sin(2 * np.pi * 12 * t + 1.0), # Different phase
        0.5 * np.sin(2 * np.pi * 2 * t)   # Low frequency modulator
    ]
    
    neural_data = np.zeros((n_time, n_neurons))
    
    # Create correlated activity across neurons
    for i in range(n_neurons):
        # Mix base signals with neuron-specific weights
        weights = np.random.dirichlet(np.ones(len(base_signals)))
        neuron_signal = sum(w * s for w, s in zip(weights, base_signals))
        
        # Add small independent noise
        neural_data[:, i] = neuron_signal + 0.1 * np.random.randn(n_time)
    
    return neural_data

def create_unconscious_like_data():
    """Create data that mimics unconscious state characteristics"""
    np.random.seed(42)
    n_time = 2000
    n_neurons = 40
    
    # Create less structured, more independent data
    neural_data = np.zeros((n_time, n_neurons))
    
    for i in range(n_neurons):
        # Each neuron has independent random activity
        neural_data[:, i] = np.random.randn(n_time)
        
        # Add some very local temporal structure
        for j in range(10, n_time, 50):
            neural_data[j:j+5, i] += 2.0  # Brief bursts
    
    return neural_data

def compare_states():
    """Compare conscious vs unconscious state signatures"""
    print("Comparing Conscious vs Unconscious Neural Signatures")
    print("="*50)
    
    # Generate data
    conscious_data = create_conscious_like_data()
    unconscious_data = create_unconscious_like_data()
    
    analyzer = NeuralCompressionAnalyzer()
    
    print("\nCONSCIOUS-LIKE STATE:")
    conscious_metrics = analyzer.compute_all_metrics(conscious_data)
    
    print("\nUNCONSCIOUS-LIKE STATE:") 
    unconscious_metrics = analyzer.compute_all_metrics(unconscious_data)
    
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    
    print("\nMetric differences (Conscious - Unconscious):")
    for metric in conscious_metrics:
        if metric in unconscious_metrics:
            diff = conscious_metrics[metric] - unconscious_metrics[metric]
            print(f"{metric:25}: {diff:+.4f}")
    
    # AEP consciousness signature
    conscious_signature = (
        conscious_metrics['intrinsic_dimensionality'] < 15 and
        conscious_metrics['predictive_complexity'] < 0.3 and
        conscious_metrics['information_integration'] > 0.05 and
        conscious_metrics['network_efficiency'] > 0.1
    )
    
    unconscious_signature = (
        unconscious_metrics['intrinsic_dimensionality'] > 20 or
        unconscious_metrics['predictive_complexity'] > 0.5 or
        unconscious_metrics['information_integration'] < 0.02
    )
    
    print(f"\nConscious signature detected: {conscious_signature}")
    print(f"Unconscious signature detected: {unconscious_signature}")

if __name__ == "__main__":
    compare_states()
