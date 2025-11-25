"""
Quantum Context Simulation
Implementation of AEP solution to quantum measurement problem
"""
import numpy as np
from scipy import linalg

class ContextMeasurement:
    """Implements AEP quantum measurement with context dependence"""
    
    def __init__(self):
        self.measurement_history = []
    
    def quantum_complexity(self, state_vector):
        """
        Estimate quantum Kolmogorov complexity of a state
        
        Parameters:
        state_vector: quantum state vector
        
        Returns:
        complexity: estimated descriptive complexity
        """
        # Simple complexity estimate based on entropy and superposition
        prob_dist = np.abs(state_vector)**2
        entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
        
        # Complexity increases with superposition and entropy
        complexity = entropy * len(state_vector)
        return complexity
    
    def superposition_complexity(self, state_vector, context):
        """
        Compute complexity of maintaining superposition in given context
        
        Parameters:
        state_vector: quantum state in superposition
        context: dictionary describing measurement context
        
        Returns:
        complexity: context-dependent superposition complexity
        """
        base_complexity = self.quantum_complexity(state_vector)
        
        # Context factors that increase complexity
        context_factors = {
            'system_size': context.get('system_size', 1),
            'environment_coupling': context.get('environment_coupling', 0),
            'measurement_precision': context.get('measurement_precision', 1),
            'apparatus_complexity': context.get('apparatus_complexity', 1)
        }
        
        # Complexity grows with context factors
        complexity_amplification = (
            context_factors['system_size'] *
            (1 + context_factors['environment_coupling']) *
            context_factors['measurement_precision'] *
            context_factors['apparatus_complexity']
        )
        
        return base_complexity * complexity_amplification
    
    def should_collapse(self, superposition, definite_states, context):
        """
        AEP measurement criterion: collapse when definite description is simpler
        
        Parameters:
        superposition: superposition state vector
        definite_states: list of possible definite states
        context: measurement context
        
        Returns:
        (should_collapse, chosen_state_index, complexity_ratio)
        """
        # Complexity of maintaining superposition
        K_sup = self.superposition_complexity(superposition, context)
        
        # Complexity of definite outcomes
        K_definite = []
        for state in definite_states:
            K_state = self.quantum_complexity(state)
            K_context = context.get('measurement_complexity', 1)
            K_definite.append(K_state + K_context)
        
        # Find simplest definite description
        best_definite_idx = np.argmin(K_definite)
        best_K_definite = K_definite[best_definite_idx]
        
        # AEP criterion: collapse if definite description is simpler
        should_collapse = best_K_definite < K_sup
        complexity_ratio = best_K_definite / K_sup if K_sup > 0 else float('inf')
        
        return should_collapse, best_definite_idx, complexity_ratio
    
    def simulate_measurement(self, initial_state, possible_outcomes, context, n_measurements=100):
        """
        Simulate multiple measurements with AEP collapse criterion
        
        Parameters:
        initial_state: initial superposition state
        possible_outcomes: list of possible measurement outcomes
        context: measurement context
        n_measurements: number of measurements to simulate
        
        Returns:
        results: dictionary with measurement statistics
        """
        collapse_decisions = []
        chosen_states = []
        complexity_ratios = []
        
        for i in range(n_measurements):
            # Add small random perturbation to context (simulating different measurement instances)
            perturbed_context = context.copy()
            perturbed_context['apparatus_complexity'] *= (0.95 + 0.1 * np.random.random())
            
            # Apply AEP collapse criterion
            should_collapse, state_idx, comp_ratio = self.should_collapse(
                initial_state, possible_outcomes, perturbed_context
            )
            
            collapse_decisions.append(should_collapse)
            chosen_states.append(state_idx)
            complexity_ratios.append(comp_ratio)
            
            self.measurement_history.append({
                'measurement_id': i,
                'collapsed': should_collapse,
                'state_index': state_idx,
                'complexity_ratio': comp_ratio,
                'context': perturbed_context
            })
        
        # Statistics
        collapse_probability = np.mean(collapse_decisions)
        state_distribution = np.bincount(chosen_states, minlength=len(possible_outcomes)) / n_measurements
        
        results = {
            'collapse_probability': collapse_probability,
            'state_distribution': state_distribution,
            'mean_complexity_ratio': np.mean(complexity_ratios),
            'collapse_decisions': collapse_decisions,
            'chosen_states': chosen_states,
            'complexity_ratios': complexity_ratios
        }
        
        return results

class QuantumSystem:
    """Simple quantum system simulator"""
    
    def __init__(self, n_states=2):
        self.n_states = n_states
        self.state = None
    
    def create_superposition(self, coefficients=None):
        """Create a superposition state"""
        if coefficients is None:
            # Equal superposition
            coefficients = np.ones(self.n_states) / np.sqrt(self.n_states)
        else:
            coefficients = np.array(coefficients)
            coefficients = coefficients / np.linalg.norm(coefficients)  # Normalize
        
        self.state = coefficients
        return self.state
    
    def get_basis_states(self):
        """Get computational basis states"""
        return [np.eye(1, self.n_states, i).flatten() for i in range(self.n_states)]

# Example usage
if __name__ == "__main__":
    print("=== AEP Quantum Measurement Simulation ===")
    
    # Create quantum system
    qsystem = QuantumSystem(n_states=3)
    superposition = qsystem.create_superposition([1, 1, 1])  # Equal superposition
    basis_states = qsystem.get_basis_states()
    
    print(f"Superposition state: {superposition}")
    print(f"Basis states: {[state.tolist() for state in basis_states]}")
    
    # Define measurement contexts
    simple_context = {
        'system_size': 1,
        'environment_coupling': 0.1,
        'measurement_precision': 1,
        'apparatus_complexity': 1,
        'measurement_complexity': 1
    }
    
    complex_context = {
        'system_size': 10,
        'environment_coupling': 0.9,
        'measurement_precision': 10,
        'apparatus_complexity': 5,
        'measurement_complexity': 3
    }
    
    # Test AEP measurement
    measurement_sim = ContextMeasurement()
    
    print("\n=== Simple Context (should maintain superposition) ===")
    results_simple = measurement_sim.simulate_measurement(
        superposition, basis_states, simple_context, n_measurements=50
    )
    print(f"Collapse probability: {results_simple['collapse_probability']:.3f}")
    print(f"State distribution: {results_simple['state_distribution']}")
    print(f"Mean complexity ratio: {results_simple['mean_complexity_ratio']:.3f}")
    
    print("\n=== Complex Context (should collapse) ===")
    results_complex = measurement_sim.simulate_measurement(
        superposition, basis_states, complex_context, n_measurements=50
    )
    print(f"Collapse probability: {results_complex['collapse_probability']:.3f}")
    print(f"State distribution: {results_complex['state_distribution']}")
    print(f"Mean complexity ratio: {results_complex['mean_complexity_ratio']:.3f}")
    
    print("\n=== Context Dependence Demonstration ===")
    # Show how collapse depends on apparatus complexity
    apparatus_complexities = np.linspace(1, 10, 10)
    collapse_probs = []
    
    for complexity in apparatus_complexities:
        context = simple_context.copy()
        context['apparatus_complexity'] = complexity
        
        results = measurement_sim.simulate_measurement(
            superposition, basis_states, context, n_measurements=20
        )
        collapse_probs.append(results['collapse_probability'])
    
    print("Apparatus Complexity vs Collapse Probability:")
    for comp, prob in zip(apparatus_complexities, collapse_probs):
        print(f"  Complexity {comp:.1f}: {prob:.3f}")
