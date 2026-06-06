import os
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg
import netket as nk
import optax
import netket.experimental

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from physics.hamiltonian import get_Hamiltonian
from physics.utils import BestIterKeeper
from physics.utils import plot_markov_autocorrelation

def run_rbm():
    print(">>> BENCHMARK 03: RBM + Metropolis")
    print("---------------------------------------------------------")
    
    N = 10
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=1.0, alpha=3.0, hilbert=hi)

    
    model = nk.models.RBM(alpha=1, param_dtype=float)


    sampler = nk.sampler.MetropolisLocal(
        hi,
        n_chains=1, # Número de exploradores en paralelo
        sweep_size=1   # Muestras que se dejan pasar entre extracciones
    )
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
    
    
    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

    print("Precalentando y compilando con JAX...")
    gs.run(n_iter=1, show_progress=False)
    jax.block_until_ready(vstate.variables)

    keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

    print("Iniciando benchmark cronometrado...")
    start_time = time.time()
    
    log = nk.logging.JsonLog("resultado_benchmark_03_RBM", save_params=False)
    gs.run(n_iter=1000, out=log, show_progress=True, callback=keeper.update)
    
    jax.block_until_ready(vstate.variables)
    end_time = time.time()
    
    print(f"\nEntrenamiento terminado. Restaurando la mejor iteración (Energia: {keeper.best_energy:.6f})...")
    vstate.parameters = keeper.best_state.parameters

    print("Calculando métricas finales...")
    E_stat = vstate.expect(H)
    E_mean = E_stat.mean.real
    E_var = E_stat.variance.real
    tau_c = getattr(E_stat, "tau_corr", 0.0)
    pearson_dev = jnp.sqrt(E_var) / abs(E_mean)

    # Cálculo Exacto
    H_sparse = H.to_sparse()
    evals, evecs = scipy.sparse.linalg.eigsh(H_sparse, k=1, which="SA")
    psi_exact = evecs[:, 0]
    E_exact = evals[0]

    psi_vmc = vstate.to_array(normalize=True)
    overlap = float(jnp.abs(jnp.vdot(psi_exact, psi_vmc))**2)
    print("Calculando el paso exacto de caída al 10%...")
    
    
    E_loc = np.array(vstate.local_estimators(H).real)[0]
    
    E_mean_chain = np.mean(E_loc)
    E_var_chain = np.var(E_loc)
    
    t_10_percent = "> Max Lag" 
    max_lag_search = min(200, len(E_loc) - 1) 
    
    for t in range(1, max_lag_search):
        
        cov_t = np.mean((E_loc[:-t] - E_mean_chain) * (E_loc[t:] - E_mean_chain))
        c_t = cov_t / E_var_chain
        
        if c_t <= 0.1:
            t_10_percent = t
            break
    print("\n>>> RESULTADOS FINALES:")
    print(f"Energia VMC       : {E_mean:.6f}")
    print(f"Energia Exacta    : {E_exact:.6f}")
    print(f"Error Relativo    : {abs((E_mean - E_exact)/E_exact):.2%}")
    print(f"Desviacion Pearson: {pearson_dev:.6f}")
    print(f"Fidelidad         : {overlap:.6f}")
    print(f"Autocorrelación τ : {tau_c:.4f}")
    print(f"Tiempo puro       : {end_time - start_time:.2f} s")
    print(f"Pasos de correlación (10%)   : {t_10_percent}")

    benchmark_title = "RBM + Metropolis"
        
    plot_markov_autocorrelation(
        vstate=vstate, 
        H=H, 
        benchmark_name=benchmark_title, 
        max_lag=40, 
        filename="autocorr_03_RBM.png" 
        ) 

if __name__ == "__main__":
    run_rbm()