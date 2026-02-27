import os
import sys
import time
import jax
import netket as nk
import optax

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from physics.hamiltonian import get_Hamiltonian

def run_ar_direct():
    print(">>> BENCHMARK 06: RED AUTOREGRESIVA (ARNN) + DIRECT SAMPLING")
    print("---------------------------------------------------------")
    
    N = 10
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=1.0, alpha=3.0, hilbert=hi)

    try:
        with open("benchmark_exact.txt", "r") as f:
            E_exact = float(f.read())
    except:
        E_exact = None

    model = nk.models.ARNNDense(
        hilbert=hi,
        layers=3,
        features=32
    )

    sampler = nk.sampler.ARDirectSampler(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
    
    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

    print("Precalentando y compilando con JAX (1 iteracion)...")
    gs.run(n_iter=1, show_progress=False)
    jax.block_until_ready(vstate.variables)

    print("Iniciando benchmark cronometrado...")
    start_time = time.time()
    
    log = nk.logging.JsonLog("resultado_benchmark_06", save_params=False)
    gs.run(n_iter=1500, out=log, show_progress=True)
    
    jax.block_until_ready(vstate.variables)
    end_time = time.time()
    
    E_final = log["Energy"].Mean[-1]
    
    print("\n>>> RESULTADOS 06_AR_DIRECT:")
    print(f"Energia AR-Direct : {E_final:.6f}")
    if E_exact:
        print(f"Energia Exacta    : {E_exact:.6f}")
        print(f"Error             : {abs((E_final - E_exact)/E_exact):.2%}")
    print(f"Tiempo de ejecucion puro: {end_time - start_time:.2f} s")

if __name__ == "__main__":
    run_ar_direct()