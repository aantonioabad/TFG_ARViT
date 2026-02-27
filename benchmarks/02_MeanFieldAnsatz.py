import os
import sys
import time
import jax
import netket as nk
import optax

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from physics.hamiltonian import get_Hamiltonian

def run_mean_field():
    print(">>> BENCHMARK 02: MEAN FIELD ANSATZ")
    print("----------------------------------------------------")
    
    N = 10
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=1.0, alpha=3.0, hilbert=hi)

    model = nk.models.RBM(alpha=0, param_dtype=float)

    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
    
    optimizer = nk.optimizer.Sgd(learning_rate=0.05)
    sr = nk.optimizer.SR(diag_shift=0.1)
    gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=sr)

    print("Precalentando y compilando con JAX...")
    gs.run(n_iter=1, show_progress=False)
    jax.block_until_ready(vstate.variables)

    print("Cronometrado...")
    start_time = time.time()
    
    log = nk.logging.JsonLog("resultado_benchmark_02", save_params=False)
    gs.run(n_iter=500, out=log, show_progress=True)
    
    jax.block_until_ready(vstate.variables)
    end_time = time.time()
    
    print(f"\nEnergia final: {log['Energy'].Mean[-1]:.6f}")
    print(f"Tiempo de ejecucion puro: {end_time - start_time:.2f} s")

if __name__ == "__main__":
    run_mean_field()