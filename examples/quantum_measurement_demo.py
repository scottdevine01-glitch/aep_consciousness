from modules.quantum_context import QuantumSystem, ContextMeasurement

# Create quantum system
qsystem = QuantumSystem(n_states=3)
superposition = qsystem.create_superposition([1, 1, 1])

# Define measurement context
context = {
    'system_size': 5,
    'environment_coupling': 0.5,
    'apparatus_complexity': 3
}

# Simulate measurement
measurement_sim = ContextMeasurement()
results = measurement_sim.simulate_measurement(
    superposition, 
    qsystem.get_basis_states(), 
    context,
    n_measurements=100
)
