# Anti-Entropic Principle (AEP) - Unified Framework

This repository contains the implementation, data, and analysis scripts for the paper:

**"The Anti-Entropic Principle: Unified Solutions to Consciousness, Quantum Foundations, and Cosmological Problems"**

by Scott Devine.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX) *<<-- (Optional but recommended: Get a DOI for your code from Zenodo)*

## Abstract

Foundational problems persist across multiple domains of science: the hard problem of consciousness, the quantum measurement problem, and the cosmological constant problem. Traditional approaches address these problems in isolation. We demonstrate that the Anti-Entropic Principle (AEP)--which states that physical laws minimize total descriptive complexity \(K(T)+K(E|T)\)--provides unified solutions across all these domains.

This repository provides the computational framework to validate the AEP's predictions in neuroscience, quantum foundations, and cosmology.

## Repository Structure

*   `01_neural_compression/`: Code to compute the six neural compression metrics (Intrinsic Dimensionality, Predictive Complexity, etc.) and reproduce the consciousness-related predictions.
*   `02_quantum_foundations/`: Simulations for context-dependent quantum collapse based on the AEP measurement criterion.
*   `03_cosmological_sims/`: Code for cosmological simulations predicting \(f_{\text{NL}}^{\text{equil}} = -0.416\), \(r < 10^{-4}\), and scale-dependent growth.
*   `04_unified_framework/`: Implementations of the unified mathematical framework and master compression equation.
*   `utilities/`: Shared helper functions and configuration.

## Installation & Requirements

1.  Clone this repository:
    ```bash
    git clone https://github.com/your-username/aep_consciousness.git
    cd aep_consciousness
    ```
2.  We recommend using a Python virtual environment.
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(You will need to create this `requirements.txt` file listing your dependencies, e.g., `numpy`, `scipy`, `matplotlib`, `scikit-learn`)*

## Usage

Each module is designed to be run independently. Please see the specific `README` file within each numbered directory for detailed instructions.

**Example: Reproducing Neural Compression Signatures**
```bash
cd 01_neural_compression/scripts
python calculate_intrinsic_dimensionality.py --input ../data/sample_fmri.npy
