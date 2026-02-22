import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# Obtiene la ruta del padre (ej: .../TFG_ARViT)
parent_dir = os.path.dirname(current_dir)
# Añade el padre al sistema de búsqueda de Python
sys.path.append(parent_dir)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["JAX_PLATFORM_NAME"] = "cpu" # O "gpu" 


import netket as nk
import numpy as np
import time
from physics.hamiltonian import get_Hamiltonian

def run_mean_field():
    print(">>> BENCHMARK 02: MEAN-FIELD (Ansatz Simple)")
    print("----------------------------------------------------------------")
    
    # 1. Sistema
    N = 10
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=1.0, alpha=3.0, hilbert=hi) # Tu Hamiltoniano
    
    # Cargar energía exacta para comparar
    try:
        with open("benchmark_exact.txt", "r") as f:
            E_exact = float(f.read())
    except:
        E_exact = None

    # 2. El Modelo: RBM con alpha=0
    model = nk.models.RBM(alpha=0, param_dtype=float)

    # 3. Sampler y Variational State
    
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)

    # 4. Optimizador
    optimizer = nk.optimizer.Sgd(learning_rate=0.05)
    sr = nk.optimizer.SR(diag_shift=0.1)
    gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=sr)

    # 5. Ejecutar
    start_time = time.time()
    log = nk.logging.JsonLog("resultado_benchmark_02", save_params=False)
    gs.run(n_iter=300, out=log, show_progress=True)
    end_time = time.time()
    
    # 6. Resultados
    E_final = log["Energy"].Mean[-1]
    
    print(f"\n>>> RESULTADOS MEAN-FIELD:")
    print(f"Energía MF    : {E_final:.6f}")
    
    if E_exact:
        print(f"Energía Exacta: {E_exact:.6f}")
        err = abs((E_final - E_exact)/E_exact)
        print(f"Error Relativo: {err:.2%}")
        
    print(f"Tiempo Total  : {end_time - start_time:.2f} s")
    
    # Explicación del resultado para ti:
    if E_exact and abs(E_final - E_exact) > 0.5:
        print("El Mean-Field no captura el entrelazamiento, así que nunca llegará a la energía exacta.")
        print("Esta es la 'base' que el Transformer debe superar.")

if __name__ == "__main__":
    run_mean_field()