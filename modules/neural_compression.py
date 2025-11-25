"""
Neural Compression Analysis
Complete implementation of the 6 compression metrics for consciousness detection
"""
import numpy as np
from scipy import linalg, stats
from scipy.linalg import toeplitz
import warnings
import sys

class NeuralCompressionAnalyzer:
    """Implements all 6 neural compression metrics from the AEP framework"""
    
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
            
            # Simple AR model prediction
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
    
    def information_integration(self, time_series, tau=1):
        """
        Simplified information integration (Φ) calculation
        Based on Tononi's IIT but framed as compression metric
        
        Parameters:
        time_series: array of shape (n_timepoints, n_channels)
        tau: time lag for mutual information
        
        Returns:
        Phi: information integration value
        """
        n_time, n_channels = time_series.shape
        
        if n_channels < 2:
            return 0.0
        
        # Binarize data for simplicity
        binary_data = (time_series > np.median(time_series, axis=0)).astype(int)
        
        # Compute whole system mutual information
        system_mi = 0.0
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                # Simple mutual information approximation
                p_ij = np.histogram2d(binary_data[:-tau, i], binary_data[tau:, j], 
                                    bins=2)[0] / (n_time - tau)
                p_i = np.histogram(binary_data[:-tau, i], bins=2)[0] / (n_time - tau)
                p_j = np.histogram(binary_data[tau:, j], bins=2)[0] / (n_time - tau)
                
                # Avoid division by zero
                mask = p_ij > 0
                if np.any(mask):
                    mi_ij = np.sum(p_ij[mask] * np.log(p_ij[mask] / (p_i.reshape(-1,1)[mask] * p_j.reshape(1,-1)[mask])))
                    system_mi += mi_ij
        
        # Simplified partition (just split in half)
        split_point = n_channels // 2
        part1_mi = 0.0
        for i in range(split_point):
            for j in range(i+1, split_point):
                # Similar MI calculation for partition 1
                p_ij = np.histogram2d(binary_data[:-tau, i], binary_data[tau:, j], 
                                    bins=2)[0] / (n_time - tau)
                p_i = np.histogram(binary_data[:-tau, i], bins=2)[0] / (n_time - tau)
                p_j = np.histogram(binary_data[tau:, j], bins=2)[0] / (n_time - tau)
                
                mask = p_ij > 0
                if np.any(mask):
                    mi_ij = np.sum(p_ij[mask] * np.log(p_ij[mask] / (p_i.reshape(-1,1)[mask] * p_j.reshape(1,-1)[mask])))
                    part1_mi += mi_ij
        
        part2_mi = 0.0
        for i in range(split_point, n_channels):
            for j in range(i+1, n_channels):
                # Similar MI calculation for partition 2
                p_ij = np.histogram2d(binary_data[:-tau, i], binary_data[tau:, j], 
                                    bins=2)[0] / (n_time - tau)
                p_i = np.histogram(binary_data[:-tau, i], bins=2)[0] / (n_time - tau)
                p_j = np.histogram(binary_data[tau:, j], bins=2)[0] / (n_time - tau)
                
                mask = p_ij > 0
                if np.any(mask):
                    mi_ij = np.sum(p_ij[mask] * np.log(p_ij[mask] / (p_i.reshape(-1,1)[mask] * p_j.reshape(1,-1)[mask])))
                    part2_mi += mi_ij
        
        # Simplified Φ calculation
        phi_value = system_mi - (part1_mi + part2_mi)
        self.metrics['information_integration'] = max(phi_value, 0)
        return max(phi_value, 0)
    
    def network_efficiency(self, neural_data):
        """
        Compute global efficiency of functional connectivity network
        
        Parameters:
        neural_data: array of shape (n_timepoints, n_nodes)
        
        Returns:
        E_global: global efficiency measure
        """
        n_time, n_nodes = neural_data.shape
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(neural_data.T)
        
        # Threshold to create adjacency matrix
        adjacency = np.abs(corr_matrix) > 0.3  # Simple threshold
        np.fill_diagonal(adjacency, 0)  # Remove self-connections
        
        # Convert to distance matrix (inverse of connectivity)
        distance_matrix = np.zeros_like(adjacency, dtype=float)
        distance_matrix[adjacency] = 1.0 / corr_matrix[adjacency]
        distance_matrix[~adjacency] = np.inf
        np.fill_diagonal(distance_matrix, 0)
        
        # Floyd-Warshall algorithm for shortest paths
        n = n_nodes
        dist = distance_matrix.copy()
        
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        # Compute global efficiency
        efficiency_sum = 0
        count = 0
        for i in range(n):
            for j in range(n):
                if i != j and dist[i][j] < np.inf:
                    efficiency_sum += 1.0 / dist[i][j]
                    count += 1
        
        e_global = efficiency_sum / count if count > 0 else 0
        self.metrics['network_efficiency'] = e_global
        return e_global
    
    def multi_scale_entropy(self, signal, max_scale=20):
        """
        Compute multi-scale entropy
        
        Parameters:
        signal: 1D time series
        max_scale: maximum scale factor
        
        Returns:
        MSE: mean sample entropy across scales
        """
        def sample_entropy(signal, m=2, r=0.2):
            """Compute sample entropy"""
            n = len(signal)
            if n <= m:
                return 0
            
            # Standard deviation-based tolerance
            tolerance = r * np.std(signal)
            
            # Count matches
            matches_m = 0
            matches_m1 = 0
            total_pairs = 0
            
            for i in range(n - m):
                for j in range(i + 1, n - m):
                    dist_m = np.max(np.abs(signal[i:i+m] - signal[j:j+m]))
                    dist_m1 = np.max(np.abs(signal[i:i+m+1] - signal[j:j+m+1]))
                    
                    if dist_m <= tolerance:
                        matches_m += 1
                        if dist_m1 <= tolerance:
                            matches_m1 += 1
                    total_pairs += 1
            
            if matches_m == 0:
                return 0
            
            return -np.log(matches_m1 / matches_m)
        
        entropy_values = []
        for scale in range(1, max_scale + 1):
            # Coarse-graining
            coarse_signal = []
            for i in range(0, len(signal) - scale + 1, scale):
                coarse_signal.append(np.mean(signal[i:i+scale]))
            
            if len(coarse_signal) > 10:  # Minimum length requirement
                se = sample_entropy(np.array(coarse_signal))
                if not np.isnan(se) and not np.isinf(se):
                    entropy_values.append(se)
        
        mse_value = np.mean(entropy_values) if entropy_values else 0
        self.metrics['multi_scale_entropy'] = mse_value
        return mse_value
    
    def compression_distance(self, signal1, signal2):
        """
        Compute compression distance between two signals
        
        Parameters:
        signal1, signal2: 1D time series
        
        Returns:
        D_comp: compression distance measure
        """
        def lz_complexity(s):
            """Simplified LZ complexity measure"""
            if len(s) == 0:
                return 0
            
            # Convert to string of symbols
            quantiles = np.percentile(s, [25, 50, 75])
            symbols = []
            for val in s:
                if val < quantiles[0]:
                    symbols.append('0')
                elif val < quantiles[1]:
                    symbols.append('1')
                elif val < quantiles[2]:
                    symbols.append('2')
                else:
                    symbols.append('3')
            
            symbol_string = ''.join(symbols)
            
            # Simplified LZ complexity count
            complexity = 0
            i = 0
            n = len(symbol_string)
            
            while i < n:
                j = i + 1
                while j <= n and symbol_string[i:j] in symbol_string[:i]:
                    j += 1
                complexity += 1
                i = j
            
            return complexity
        
        # Compute individual complexities
        c1 = lz_complexity(signal1)
        c2 = lz_complexity(signal2)
        
        # Compute joint complexity (concatenate signals)
        joint_signal = np.concatenate([signal1, signal2])
        c_joint = lz_complexity(joint_signal)
        
        # Normalize
        c1_norm = c1 / len(signal1) if len(signal1) > 0 else 0
        c2_norm = c2 / len(signal2) if len(signal2) > 0 else 0
        c_joint_norm = c_joint / len(joint_signal) if len(joint_signal) > 0 else 0
        
        # Compression distance
        d_comp = (c1_norm - c_joint_norm) / c1_norm if c1_norm > 0 else 0
        self.metrics['compression_distance'] = d_comp
        return d_comp
    
    def compute_all_metrics(self, neural_data):
        """Compute all 6 compression metrics"""
        print("Computing neural compression metrics...")
        
        # Intrinsic Dimensionality
        id_val = self.intrinsic_dimensionality(neural_data)
        print(f"1. Intrinsic Dimensionality: {id_val}")
        
        # Predictive Complexity  
        pc_val = self.predictive_complexity(neural_data)
        print(f"2. Predictive Complexity: {pc_val:.4f}")
        
        # Information Integration
        phi_val = self.information_integration(neural_data)
        print(f"3. Information Integration (Φ): {phi_val:.4f}")
        
        # Network Efficiency
        eff_val = self.network_efficiency(neural_data)
        print(f"4. Network Efficiency: {eff_val:.4f}")
        
        # Multi-scale Entropy (using first channel)
        mse_val = self.multi_scale_entropy(neural_data[:, 0])
        print(f"5. Multi-scale Entropy: {mse_val:.4f}")
        
        # Compression Distance (using first two channels)
        if neural_data.shape[1] >= 2:
            comp_val = self.compression_distance(neural_data[:, 0], neural_data[:, 1])
            print(f"6. Compression Distance: {comp_val:.4f}")
        else:
            comp_val = 0
            print("6. Compression Distance: N/A (need at least 2 channels)")
        
        return self.metrics

# Example usage and testing
if __name__ == "__main__":
    # Generate sample neural data
    np.random.seed(42)
    
    # Create structured data (simulating conscious state)
    n_time = 1000
    n_neurons = 50
    
    # Base signal with some structure
    t = np.linspace(0, 10, n_time)
    base_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
    
    # Create correlated neural data
    neural_data = np.zeros((n_time, n_neurons))
    for i in range(n_neurons):
        neural_data[:, i] = base_signal * (0.8 + 0.4 * np.random.random()) + 0.1 * np.random.randn(n_time)
    
    print("=== AEP Neural Compression Analysis ===")
    analyzer = NeuralCompressionAnalyzer()
    metrics = analyzer.compute_all_metrics(neural_data)
    
    print("\n=== Summary ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
