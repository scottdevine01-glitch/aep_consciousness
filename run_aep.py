#!/usr/bin/env python3
"""
AEP Framework - Run without installation
"""
import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def check_dependencies():
    """Check and install required packages"""
    required_packages = [
        'numpy', 'scipy', 'pandas', 'matplotlib', 
        'sklearn', 'networkx', 'astropy'
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

def main():
    print("AEP Framework - Running without installation")
    
    if check_dependencies():
        print("\nAll dependencies satisfied!")
        print("You can now import modules from the modules/ directory")
        
        # Example usage
        try:
            from modules import neural_compression
            print("Neural compression module loaded successfully")
        except ImportError as e:
            print(f"Import error: {e}")
    else:
        print("\nPlease install missing dependencies first")

if __name__ == "__main__":
    main()
