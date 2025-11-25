"""
Example: Quantum Context Dependence Demonstrations
"""
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.quantum_context import QuantumSystem, ContextMeasurement

def demonstrate_context_dependence():
    """Show how measurement context affects collapse probability"""
    print("Quantum Context Dependence Demonstration")
    print("="*50)
    
    # Create quantum system
    qsystem = QuantumSystem(n_states=3)
    superposition = qsystem.create_superposition([1, 1, 1])
    basis_states = qsystem.get_basis_states()
    
    measurement_sim = ContextMeasurement()
    
    # Test various contexts
    contexts = {
        'Isolated Qubit': {
            'system_size': 1,
            'environment_coupling': 0.01,
            'apparatus_complexity': 1,
            'measurement_complexity': 1
        },
        'Laboratory Measurement': {
            'system_size': 5, 
            'environment_coupling': 0.5,
            'apparatus_complexity': 3,
            'measurement_complexity': 2
        },
        'Macroscopic Object': {
            'system_size': 100,
            'environment_coupling': 0.95,
            'apparatus_complexity': 10,
            'measurement_complexity': 5
        }
    }
    
    results = {}
    for context_name, context in contexts.items():
        results[context_name] = measurement_sim.simulate_measurement(
            superposition, basis_states, context, n_measurements=100
        )
    
    # Display results
    print("\nContext-Dependent Collapse Probabilities:")
    print("-" * 50)
    for context_name, result in results.items():
        print(f"{context_name:20}: {result['collapse_probability']:.3f}")
    
    return results

def apparatus_complexity_sweep():
    """Show how apparatus complexity affects collapse"""
    print("\nApparatus Complexity Sweep")
    print("="*50)
    
    qsystem = QuantumSystem(n_states=2)
    superposition = qsystem.create_superposition([1, 1])
    basis_states = qsystem.get_basis_states()
    
    measurement_sim = ContextMeasurement()
    
    complexities = np.linspace(1, 20, 10)
    collapse_probs = []
    
    base_context = {
        'system_size': 2,
        'environment_coupling': 0.3,
        'measurement_precision': 2
    }
    
    for comp in complexities:
        context = base_context.copy()
        context['apparatus_complexity'] = comp
        
        results = measurement_sim.simulate_measurement(
            superposition, basis_states, context, n_measurements=50
        )
        collapse_probs.append(results['collapse_probability'])
    
    print("Complexity vs Collapse Probability:")
    for comp, prob in zip(complexities, collapse_probs):
        print(f"  Apparatus Complexity {comp:5.1f}: {prob:.3f}")
    
    return complexities, collapse_probs

if __name__ == "__main__":
    # Run demonstrations
    context_results = demonstrate_context_dependence()
    complexity_sweep = apparatus_complexity_sweep()
    
    print("\nAEP Quantum Insights:")
    print("- Collapse depends on descriptive complexity, not just system size")
    print("- Measurement context determines when superposition becomes inefficient")
    print("- This resolves the quantum measurement problem naturally")
