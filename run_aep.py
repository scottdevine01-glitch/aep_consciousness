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
            metrics['predictive_complexity'] < 0.5 and
            metrics['information_integration'] > 0.1
        )
        print(f"Conscious-like signature: {conscious_signature}")
        
    except ImportError as e:
        print(f"Neural module import error: {e}")

def run_quantum_demo():
    """Run quantum measurement demo"""
    print("\n" + "="*50)
    print("QUANTUM MEASUREMENT DEMO")
    print("="*50)
    
    try:
        from quantum_context import QuantumSystem, ContextMeasurement
        
        # Create quantum system
        qsystem = QuantumSystem(n_states=2)
        superposition = qsystem.create_superposition([1, 1])  # Equal superposition
        basis_states = qsystem.get_basis_states()
        
        print(f"Superposition state: {superposition}")
        
        # Test different measurement contexts
        measurement_sim = ContextMeasurement()
        
        contexts = {
            'Simple (Microscopic)': {
                'system_size': 1, 'environment_coupling': 0.1, 
                'apparatus_complexity': 1, 'measurement_complexity': 1
            },
            'Complex (Macroscopic)': {
                'system_size': 10, 'environment_coupling': 0.9,
                'apparatus_complexity': 5, 'measurement_complexity': 3
            }
        }
        
        for context_name, context in contexts.items():
            results = measurement_sim.simulate_measurement(
                superposition, basis_states, context, n_measurements=30
            )
            print(f"\n{context_name} Context:")
            print(f"  Collapse probability: {results['collapse_probability']:.3f}")
            print(f"  State distribution: {results['state_distribution']}")
            print(f"  Complexity ratio: {results['mean_complexity_ratio']:.3f}")
            
    except ImportError as e:
        print(f"Quantum module import error: {e}")

def run_cosmology_demo():
    """Run cosmological fitting demo"""
    print("\n" + "="*50)
    print("COSMOLOGICAL FITTING DEMO")
    print("="*50)
    
    try:
        from cosmology_fitting import AEPCosmology
        
        cosmo = AEPCosmology()
        
        # Observational constraints
        observations = {
            'H0_measured': 73.0,
            'H0_error': 1.0,
            'cmb_amplitude': 1.0,
            'sigma_8_measured': 0.81,
            'f_nl_measured': -0.4
        }
        
        # Initial guess
        initial_guess = {
            'H0': 70.0,
            'Omega_m': 0.3,
            'Omega_lambda': 0.7,
            'r': 0.05,
            'f_nl': -0.5
        }
        
        print("Optimizing cosmological parameters...")
        result = cosmo.optimize_parameters(initial_guess, observations)
        
        if result.success:
            print("\nOptimal parameters found:")
            for param, value in result.optimal_params.items():
                print(f"  {param}: {value:.4f}")
            
            # AEP specific predictions
            print(f"\nAEP predictions:")
            print(f"  f_nl: {result.optimal_params['f_nl']:.3f} (target: -0.416)")
            print(f"  r: {result.optimal_params['r']:.6f} (target: < 0.0001)")
        else:
            print("Optimization failed")
            
    except ImportError as e:
        print(f"Cosmology module import error: {e}")

def main():
    print("AEP Framework - Unified Consciousness, Quantum, and Cosmology")
    print("="*60)
    
    if check_dependencies():
        print("\n" + "="*60)
        print("RUNNING DEMONSTRATIONS")
        print("="*60)
        
        # Run all demos
        run_neural_demo()
        run_quantum_demo() 
        run_cosmology_demo()
        
        print("\n" + "="*60)
        print("DEMONSTRATIONS COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("1. Use the modules in your own code")
        print("2. Import and extend the classes")
        print("3. Add your own data and experiments")
        print("4. See examples/ for more usage patterns")
    else:
        print("\nPlease install missing dependencies first")

if __name__ == "__main__":
    main()
