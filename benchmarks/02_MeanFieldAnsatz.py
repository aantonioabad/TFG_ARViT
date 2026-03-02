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

from physics.hamiltonian import get_Hamiltonian

def run_mean_field():
    print(">>> BENCHMARK 02: MEAN FIELD ANSATZ")
    print("---------------------------------------------------------")
    
    N = 10
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=1.0, alpha=3.0, hilbert=hi)

    # Forzamos param_dtype a float para evitar conflictos complejos
    model = nk.models.Jastrow(param_dtype=float)

    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
    
    optimizer = optax.adam(learning_rate=0.01)
    gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=nk.optimizer.SR(diag_shift=0.1))

    print("Precalentando y compilando con JAX (1 iteracion)...")
    gs.run(n_iter=1, show_progress=False)
    jax.block_until_ready(vstate.variables)

    print("Iniciando benchmark cronometrado...")
    start_time = time.time()
    
    log = nk.logging.JsonLog("resultado_benchmark_02", save_params=False)
    gs.run(n_iter=1500, out=log, show_progress=True)
    
    jax.block_until_ready(vstate.variables)
    end_time = time.time()
    
    print("\nCalculando metricas finales (Fidelidad y Pearson)...")
    E_stat = vstate.expect(H)
    E_mean = E_stat.mean.real
    E_var = E_stat.variance.real
    pearson_dev = jnp.sqrt(E_var) / abs(E_mean)

    H_sparse = H.to_sparse()
    evals, evecs = scipy.sparse.linalg.eigsh(H_sparse, k=1, which="SA")
    psi_exact = evecs[:, 0]
    E_exact = evals[0]

    psi_vmc = vstate.to_array(normalize=True)
    overlap = float(jnp.abs(jnp.vdot(psi_exact, psi_vmc))**2)

    print("\n>>> RESULTADOS FINALES:")
    print(f"Energia VMC       : {E_mean:.6f}")
    print(f"Energia Exacta    : {E_exact:.6f}")
    print(f"Error Relativo    : {abs((E_mean - E_exact)/E_exact):.2%}")
    print(f"Desviacion Pearson: {pearson_dev:.6f}")
    print(f"Fidelidad         : {overlap:.6f}")
    print(f"Tiempo puro       : {end_time - start_time:.2f} s")

if __name__ == "__main__":
    run_mean_field()