import os
import sys
import time
import jax
import jax.numpy as jnp
import scipy.sparse.linalg
import netket as nk
import optax
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- IMPORTACIONES LOCALES ---
from models.vitB import ARSpinViT_Causal
from physics.utils import BestIterKeeper 
from physics.utils import plot_markov_autocorrelation

import numpy as np

def get_Hamiltonian_2D(Lx: int, Ly: int, J: float, alpha: float, h: float = 1.0, hilbert=None):
    N = Lx * Ly
    if hilbert is None:
        hilbert = nk.hilbert.Spin(s=0.5, N=N)
    
    graph = nk.graph.Grid(extent=[Lx, Ly], pbc=True)
    distances = graph.distances()
    H = nk.operator.LocalOperator(hilbert)
    
    # Usamos Numpy estricto para las matrices base
    sigmax = np.array([[0, 1], [1, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])
    
    # Término de campo transversal
    for i in range(N):
        H += nk.operator.LocalOperator(hilbert, -h * sigmax, [i])
        
    # Término de interacción de largo alcance 2D
    for i in range(N):
        for j in range(i + 1, N):
            dist = distances[i][j]
            if dist > 0:
                coupling = J / (dist ** alpha)
                # [FIX]: Usamos np.kron en lugar de jnp.kron para mantenerlo en CPU
                term = coupling * np.kron(sigmaz, sigmaz)
                H += nk.operator.LocalOperator(hilbert, term, [i, j])
                
    return H

# ==============================================================================
# SCRIPT DE BENCHMARK
# ==============================================================================
def run_arvit_direct_2d():
    print(">>> BENCHMARK 2D: ARViT + DIRECT SAMPLING (Malla 4x4)")
    print("---------------------------------------------------------")
    
    Lx, Ly = 4, 4
    N = Lx * Ly
    J_val = 1.0
    alpha_val = 3.0
    
    hi = nk.hilbert.Spin(s=0.5, N=N)
    
    # Usamos la nueva función del Hamiltoniano 2D definida arriba
    H = get_Hamiltonian_2D(Lx=Lx, Ly=Ly, J=J_val, alpha=alpha_val, hilbert=hi)

    model = ARSpinViT_Causal(
        hilbert=hi,
        embedding_d=16,
        n_heads=2,
        n_blocks=2,
        n_ffn_layers=1
    )

    sampler = nk.sampler.ARDirectSampler(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
    
    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

    print("Precalentando y compilando con JAX...")
    gs.run(n_iter=1, show_progress=False)
    jax.block_until_ready(vstate.variables)

    keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

    print("Iniciando benchmark cronometrado de 500 iteraciones...")
    start_time = time.time()
    
    log = nk.logging.JsonLog("resultado_benchmark_2D_ARViT", save_params=False)
    
    gs.run(n_iter=500, out=log, show_progress=True, callback=keeper.update)
    
    jax.block_until_ready(vstate.variables)
    end_time = time.time()
    
    print(f"\nEntrenamiento terminado. Restaurando la mejor iteración (Energia: {keeper.best_energy:.6f})...")
    vstate.parameters = keeper.best_state.parameters

    # --- CÁLCULO DE NUEVAS MÉTRICAS ---
    print("Calculando métricas finales...")
    E_stat = vstate.expect(H)
    E_mean = E_stat.mean.real
    E_var = E_stat.variance.real
    tau_c = getattr(E_stat, "tau_corr", 0.0) 
    pearson_dev = jnp.sqrt(E_var) / abs(E_mean)

    print("Diagonalizando matriz exacta (ED) para N=16. Esto tomará unos segundos...")
    H_sparse = H.to_sparse()
    evals, evecs = scipy.sparse.linalg.eigsh(H_sparse, k=1, which="SA")
    psi_exact = evecs[:, 0]
    E_exact = evals[0]

    psi_vmc = vstate.to_array(normalize=True)
    overlap = float(jnp.abs(jnp.vdot(psi_exact, psi_vmc))**2)

    # --- RESULTADOS FINALES ---
    print("\n>>> RESULTADOS FINALES 2D:")
    print(f"Energia VMC       : {E_mean:.6f}")
    print(f"Energia Exacta    : {E_exact:.6f}")
    print(f"Error Relativo    : {abs((E_mean - E_exact)/E_exact):.2%}")
    print(f"Desviacion Pearson: {pearson_dev:.6f}")
    print(f"Fidelidad         : {overlap:.6f}")
    print(f"Autocorrelación τ : {tau_c:.4f}")
    print(f"Tiempo puro       : {end_time - start_time:.2f} s")

    benchmark_title = "ARViT Direct 2D (4x4)"
    plot_markov_autocorrelation(
        vstate=vstate, 
        H=H, 
        benchmark_name=benchmark_title, 
        max_lag=40, 
        filename="autocorr_2D_ARViTDirect.png" 
    )

if __name__ == "__main__":
    run_arvit_direct_2d()