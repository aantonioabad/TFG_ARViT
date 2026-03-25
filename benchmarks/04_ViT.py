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

from physics.hamiltonian import get_Hamiltonian
from physics.utils import BestIterKeeper
from physics.utils import plot_markov_autocorrelation

# Asegúrate de que la importación viene del archivo correcto donde tengas BatchedSpinViT
from models.vit_standard import BatchedSpinViT 

def run_vit_metropolis():
    print(">>> BENCHMARK 04: ViT ESTÁNDAR + METROPOLIS SAMPLING")
    print("---------------------------------------------------------")
    
    N = 10
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=1.0, alpha=3.0, hilbert=hi)

    # TU MODELO ORIGINAL EXACTO
    model = BatchedSpinViT(
        token_size=1,
        embedding_d=8,
        n_heads=2,
        n_blocks=2,
        n_ffn_layers=1,
        final_architecture=(8, 4), 
        is_complex=False
    )

    # Metropolis es lo que nos dará una autocorrelación > 0
    sampler = nk.sampler.MetropolisLocal(
        hi,
        n_chains=16, # Número de exploradores en paralelo
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
    
    log = nk.logging.JsonLog("resultado_benchmark_04_ViT", save_params=False)
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
    print(f"Autocorrelación τ : {tau_c:.4f}")
    print(f"Tiempo puro       : {end_time - start_time:.2f} s")
    plot_markov_autocorrelation(vstate, H, max_lag=40, filename="autocorr_04_ViT.png")

if __name__ == "__main__":
    run_vit_metropolis()