#!/usr/bin/env python3
"""
AEP Framework - Unified Framework for Consciousness, Quantum Physics, and Cosmology
Run without installation
"""
import sys
import os
import numpy as np

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def check_dependencies():
    """Check and install required packages"""
    required_packages = [
        'numpy', 'scipy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} missing")
    
    if missing:
        print(f"\nMissing packages: {missing}")
        print("Please install using: pip install " + " ".join(missing))
        return False
    return True

def run_neural_demo():
    """Run neural compression analysis demo"""
    print("\n" + "="*50)
    print("NEURAL COMPRESSION DEMO")
    print("="*50)
    
    try:
        from neural_compression import NeuralCompressionAnalyzer
        
        # Generate sample neural data
        np.random.seed(42)
        n_time = 1000
        n_neurons = 30
        
        # Create structured data (simulating conscious state)
        t = np.linspace(0, 10, n_time)
        base_signal = np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 12 * t)
        
        neural_data = np.zeros((n_time, n_neurons))
        for i in range(n_neurons):
            neural_data[:, i] = base_signal * (0.7 + 0.6 * np.random.random()) + 0.2 * np.random.randn(n_time)
        
        analyzer = NeuralCompressionAnalyzer()
        metrics = analyzer.compute_all_metrics(neural_data)
        
        print("\nConsciousness Signature Summary:")
        conscious_signature = (
            metrics['intrinsic_dimensionality'] < 10 and
            metrics['predictive_complexity'] < 
