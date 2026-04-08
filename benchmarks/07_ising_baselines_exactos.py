import os
import sys
import scipy.sparse.linalg

# Configuración de rutas para importar tu física
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from physics.ising_netket import get_native_Ising

def calcular_energia_exacta(N, J, dimensions, phase_name):
    print(f"\n{'-'*50}")
    print(f">>> MODELO ISING {dimensions}D | FASE: {phase_name} | N={N} | J={J}")
    
    # 1. Obtenemos el Hamiltoniano básico (Vecinos más cercanos)
    hi, H = get_native_Ising(N=N, J=J, h_x=1.0, dimensions=dimensions)
    
    # 2. Convertimos a matriz dispersa (sparse) para que quepa en la memoria
    H_sparse = H.to_sparse()
    
    # 3. Diagonalización Exacta (Sacamos el autovalor más pequeño = Energía Fundamental)
    evals, evecs = scipy.sparse.linalg.eigsh(H_sparse, k=1, which="SA")
    E_exact = evals[0]
    
    print(f"    Energía Exacta (Truth) : {E_exact:.6f}")
    print(f"    Energía por espín      : {E_exact / N:.6f}")
    print(f"{'-'*50}")

if __name__ == "__main__":
    print("\nCALCULANDO BASELINES EXACTOS PARA LA MEMORIA DEL TFG...\n")
    
    # --- MODELO 1D (Cadena de 10 espines) ---
    calcular_energia_exacta(N=10, J=-1.0, dimensions=1, phase_name="Ferromagnética (FM)")
    calcular_energia_exacta(N=10, J= 1.0, dimensions=1, phase_name="Antiferromagnética (AFM)")

    # --- MODELO 2D (Cuadrícula de 4x4 = 16 espines) ---
    calcular_energia_exacta(N=16, J=-1.0, dimensions=2, phase_name="Ferromagnética (FM)")
    calcular_energia_exacta(N=16, J= 1.0, dimensions=2, phase_name="Antiferromagnética (AFM)")
    