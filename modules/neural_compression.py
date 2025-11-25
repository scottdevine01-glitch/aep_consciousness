"""
Neural Compression Analysis
Implementation of the 6 compression metrics for consciousness detection
"""
import numpy as np
from scipy import linalg
import warnings

class NeuralCompressionAnalyzer:
    """Implements the 6 neural compression metrics from the AEP framework"""
    
    def __init__(self):
        self.metrics = {}
    
    def intrinsic_dimensionality(self, neural_data):
        """
        Compute intrinsic dimensionality of neural activity patterns
        
        Parameters:
        neural_data: array of shape (n_timepoints, n_neurons)
        
        Returns:
        ID: intrinsic dimensionality value
        """
        # Center the data
        data_centered = neural_data - np.mean(neural_data, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(data_centered, rowvar=False)
        
        # Compute eigenvalues
        eigenvalues = linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        # Normalize eigenvalues
        eigen_norm = eigenvalues / np.sum(eigenvalues)
        
        # Find number of dimensions needed to explain 95% of variance
        cumulative_variance = np.cumsum(eigen_norm)
        id_value = np.argmax(cumulative_variance >= 0.95) + 1
        
        self.metrics['intrinsic_dimensionality'] = id_value
        return id_value
    
    def predictive_complexity(self, time_series, order=6):
        """
        Compute predictive complexity via autoregressive modeling
        
        Parameters:
        time_series: array of shape (n_timepoints, n_channels)
        order: order of AR model (default=6)
        
        Returns:
        PC: predictive complexity (mean squared prediction error)
        """
        n_time, n_channels = time_series.shape
        prediction_errors = []
        
        for channel in range(n_channels):
            series = time_series[:, channel]
            
            # Simple AR model prediction (simplified)
            predictions = np.zeros_like(series)
            for t in range(order, len(series)):
                # Use previous 'order' points to predict next point
                predictions[t] = np.mean(series[t-order:t])
            
            # Compute prediction error (skip first 'order' points)
            error = np.mean((series[order:] - predictions[order:])**2)
            prediction_errors.append(error)
        
        pc_value = np.mean(prediction_errors)
        self.metrics['predictive_complexity'] = pc_value
        return pc_value
    
    def multi_scale_entropy(self, signal, max_scale=20):
        """
        Compute multi-scale entropy (simplified version)
        
        Parameters:
        signal: 1D time series
        max_scale: maximum scale factor
        
        Returns:
        MSE: mean sample entropy across scales
        """
        def sample_entropy(signal, m=2, r=0.2):
            """Compute sample entropy (simplified)"""
            n = len(signal)
            if n <= m:
                return 0
            
            # Standard deviation-based tolerance
            tolerance = r * np.std(signal)
            
            # Count matches (simplified calculation)
            matches = 0
            total = 0
            
            for i in range(n - m):
                for j in range(i + 1, n - m):
                    if np.max(np.abs(signal[i:i+m] - signal[j:j+m])) <= tolerance:
                        matches += 1
                    total += 1
            
            return -np.log(matches / total) if matches > 0 else 0
        
        entropy_values = []
        for scale in range(1, max_scale + 1):
            # Coarse-graining
            coarse_signal = []
            for i in range(0, len(signal) - scale + 1, scale):
                coarse_signal.append(np.mean(signal[i:i+scale]))
            
            if len(coarse_signal) > 10:  # Minimum length requirement
                entropy_values.append(sample_entropy(np.array(coarse_signal)))
        
        mse_value = np.mean(entropy_values) if entropy_values else 0
        self.metrics['multi_scale_entropy'] = mse_value
        return mse_value
    
    def compute_all_metrics(self, neural_data):
        """Compute all 6 compression metrics"""
        print("Computing neural compression metrics...")
        
        # Intrinsic Dimensionality
        id_val = self.intrinsic_dimensionality(neural_data)
        print(f"Intrinsic Dimensionality: {id_val}")
        
        # Predictive Complexity  
        pc_val = self.predictive_complexity(neural_data)
        print(f"Predictive Complexity: {pc_val:.4f}")
        
        # Multi-scale Entropy (using first channel for demo)
        mse_val = self.multi_scale_entropy(neural_data[:, 0])
        print(f"Multi-scale Entropy: {mse_val:.4f}")
        
        return self.metrics

# Example usage
if __name__ == "__main__":
    # Generate sample neural data
    np.random.seed(42)
    sample_data = np.random.randn(1000, 50)  # 1000 timepoints, 50 neurons
    
    analyzer = NeuralCompressionAnalyzer()
    metrics = analyzer.compute_all_metrics(sample_data)
