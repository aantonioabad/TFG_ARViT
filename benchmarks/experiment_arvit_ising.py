import os
import sys
import time
import jax
import scipy.sparse.linalg
import netket as nk
import optax

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


from physics.ising_netket import get_native_Ising
from physics.utils import BestIterKeeper, plot_markov_autocorrelation
from models.vitB import ARSpinViT_Causal 

def run_arvit_ising_experiment(N, J, dimensions, phase_name):
    print(f"\n{'='*65}")
    print(f">>> EXPERIMENTO ARViT: Ising {dimensions}D | Fase: {phase_name} | N={N} | J={J}")
    print(f"{'='*65}")

    # 1. Obtenemos el Hilbert y el Hamiltoniano nativo
    hi, H = get_native_Ising(N=N, J=J, h_x=1.0, dimensions=dimensions)

    # 2. INSTANCIAMOS TU MODELO CAUSAL
    # (Ajusta los hiperparámetros según la versión final de tu modelo)
    model = ARSpinViT_Causal(
        hilbert=hi, # Los modelos AR suelen necesitar conocer el espacio de Hilbert
        # Añade aquí tus parámetros (token_size, n_heads, n_blocks, etc.)
    )

    sampler = nk.sampler.ARDirectSampler(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
    
   
    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

    print("Precalentando y compilando con JAX...")
    gs.run(n_iter=1, show_progress=False)
    jax.block_until_ready(vstate.variables)

    keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

    print(f"Iniciando entrenamiento de ARViT para {phase_name} {dimensions}D...")
    start_time = time.time()
    
    log = nk.logging.JsonLog(f"resultado_ARViT_{dimensions}D_{phase_name}", save_params=False)
    gs.run(n_iter=500, out=log, show_progress=True, callback=keeper.update)
    
    jax.block_until_ready(vstate.variables)
    end_time = time.time()
    
    print(f"\nRestaurando la mejor iteración (Energía: {keeper.best_energy:.6f})...")
    vstate.parameters = keeper.best_state.parameters

    # 6. Cálculo de métricas
    E_stat = vstate.expect(H)
    E_mean = E_stat.mean.real
    E_var = E_stat.variance.real
    tau_c = getattr(E_stat, "tau_corr", 0.0)
    pearson_dev = jax.numpy.sqrt(E_var) / abs(E_mean)

  
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
    
    # 7. GENERACIÓN DE LA GRÁFICA DE AUTOCORRELACIÓN
    filename = f"autocorr_ARViT_{dimensions}D_{phase_name}.png"
    plot_markov_autocorrelation(vstate, H, max_lag=40, filename=filename)
    print(f"{'='*65}\n")

if __name__ == "__main__":
    
    # --- MODELO 1D (Cadena de 10 espines) ---
    run_arvit_ising_experiment(N=10, J=-1.0, dimensions=1, phase_name="FM")
    run_arvit_ising_experiment(N=10, J=1.0,  dimensions=1, phase_name="AFM")

    # --- MODELO 2D (Cuadrícula de 4x4 = 16 espines) ---
   # run_arvit_ising_experiment(N=16, J=-1.0, dimensions=2, phase_name="FM")
   # run_arvit_ising_experiment(N=16, J=1.0,  dimensions=2, phase_name="AFM")