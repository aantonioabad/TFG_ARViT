import os
import sys
import time
import jax
import jax.numpy as jnp
import scipy.sparse.linalg
import netket as nk
import optax

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Importamos tu Hamiltoniano y tus utilidades, nada más.
from physics.ising_netket import get_native_Ising
from physics.utils import BestIterKeeper

def run_pure_netket_experiment(N, J, dimensions, phase_name):
    print(f"\n{'='*60}")
    print(f">>> EXPERIMENTO NETKET PURO: Ising {dimensions}D | Fase: {phase_name} | N={N} | J={J}")
    print(f"{'='*60}")

    # 1. Obtenemos el Hilbert y el Hamiltoniano nativo de NetKet
    hi, H = get_native_Ising(N=N, J=J, h_x=1.0, dimensions=dimensions)

    # 2. USAMOS EL MODELO INTEGRADO DE NETKET (ARNN Densa)
    model = nk.models.ARNNDense(
        hilbert=hi,
        layers=3,
        features=16
    )

    # 3. Muestreo Directo integrado
    sampler = nk.sampler.ARDirectSampler(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
    
    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

    print("Precalentando y compilando con JAX...")
    gs.run(n_iter=1, show_progress=False)
    jax.block_until_ready(vstate.variables)

    # 4. Guardián de iteraciones
    keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

    print(f"Iniciando entrenamiento para {phase_name} {dimensions}D...")
    start_time = time.time()
    
    log = nk.logging.JsonLog(f"resultado_ising_nativo_{dimensions}D_{phase_name}", save_params=False)
    # 500 iteraciones para no eternizarnos, puedes subirlo a 1000 luego
    gs.run(n_iter=500, out=log, show_progress=True, callback=keeper.update)
    
    jax.block_until_ready(vstate.variables)
    end_time = time.time()
    
    print(f"\nRestaurando la mejor iteración (Energia: {keeper.best_energy:.6f})...")
    vstate.parameters = keeper.best_state.parameters

    # 5. Extracción de métricas
    E_stat = vstate.expect(H)
    E_mean = E_stat.mean.real
    E_var = E_stat.variance.real
    tau_c = getattr(E_stat, "tau_corr", 0.0)
    pearson_dev = jnp.sqrt(E_var) / abs(E_mean)

    # Diagonalización Exacta
    H_sparse = H.to_sparse()
    evals, evecs = scipy.sparse.linalg.eigsh(H_sparse, k=1, which="SA")
    E_exact = evals[0]

    print("\n>>> RESULTADOS FINALES <<<")
    print(f"Energia VMC       : {E_mean:.6f}")
    print(f"Energia Exacta    : {E_exact:.6f}")
    print(f"Error Relativo    : {abs((E_mean - E_exact)/E_exact):.2%}")
    print(f"Desviacion Pearson: {pearson_dev:.6f}")
    print(f"Autocorrelación τ : {tau_c:.4f}")
    print(f"Tiempo puro       : {end_time - start_time:.2f} s")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # --- 1D CHAIN (N=10) ---
    run_pure_netket_experiment(N=10, J=-1.0, dimensions=1, phase_name="FM")
    run_pure_netket_experiment(N=10, J=1.0,  dimensions=1, phase_name="AFM")

    # --- 2D GRID (N=16, cuadrado de 4x4) ---
    run_pure_netket_experiment(N=16, J=-1.0, dimensions=2, phase_name="FM")
    run_pure_netket_experiment(N=16, J=1.0,  dimensions=2, phase_name="AFM")