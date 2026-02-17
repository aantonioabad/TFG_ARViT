import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# Obtiene la ruta del padre (ej: .../TFG_ARViT)
parent_dir = os.path.dirname(current_dir)
# Añade el padre al sistema de búsqueda de Python
sys.path.append(parent_dir)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu" # O "gpu" 


import netket as nk
import numpy as np
import time
from physics.hamiltonian import get_Hamiltonian

def run_exact():
    print(">>> BENCHMARK 01: DIAGONALIZACIÓN EXACTA")
    
    # 1. Definir el Sistema (Igual que en tu Transformer)
    N = 10
    J = 1.0
    alpha = 3.0
    
    # Espacio de Hilbert (Spin 1/2)
    hi = nk.hilbert.Spin(s=0.5, N=N)
    
    # Hamiltoniano (Tu versión de largo alcance)
    H = get_Hamiltonian(N, J, alpha, hi)
    
    # 2. Diagonalización Exacta (Lanczos)
    # Convierte el operador H en una matriz dispersa y busca el autovalor más bajo.
    print(f"Calculando suelo exacto para N={N}...")
    start_time = time.time()
    
    result = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)
    energy_exact = result[0]
    
    end_time = time.time()
    
    print(f"\nResultados Exactos:")
    print(f"-------------------")
    print(f"Energía: {energy_exact:.6f}")
    print(f"Tiempo : {end_time - start_time:.4f} s")
    
    # Guardar en un fichero para comparar luego
    with open("benchmark_exact.txt", "w") as f:
        f.write(f"{energy_exact}")

if __name__ == "__main__":
    run_exact()