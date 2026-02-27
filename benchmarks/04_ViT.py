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
from models.vit_standard import BatchedSpinViT

def run_vit_benchmark():
    print(">>> BENCHMARK 04: ViT ESTANDAR")
    print(">>> Metodo: Ansatz No-Autoregresivo + Metropolis Sampling")
    print("---------------------------------------------------------")
    
    N = 10
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=1.0, alpha=3.0, hilbert=hi)

    try:
        with open("benchmark_exact.txt", "r") as f:
            E_exact = float(f.read())
    except:
        E_exact = None

    model = BatchedSpinViT(
        token_size=1,
        embedding_d=8,
        n_heads=2,
        n_blocks=2,
        n_ffn_layers=1,
        final_architecture=(8, 4), 
        is_complex=False
    )
    
    sampler = nk.sampler.MetropolisLocal(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)

    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

    print("Precalentando y compilando con JAX (1 iteracion)...")
    gs.run(n_iter=1, show_progress=False)
    jax.block_until_ready(vstate.variables)

    print("Iniciando benchmark cronometrado...")
    start_time = time.time()
    
    log = nk.logging.JsonLog("resultado_benchmark_04", save_params=False)
    gs.run(n_iter=500, out=log, show_progress=True)
    
    jax.block_until_ready(vstate.variables)
    end_time = time.time()
    
    E_final = log["Energy"].Mean[-1]
    
    print("\n>>> RESULTADOS 04_ViT:")
    print(f"Energia ViT : {E_final:.6f}")
    if E_exact:
        print(f"Energia Exacta: {E_exact:.6f}")
        print(f"Error       : {abs((E_final - E_exact)/E_exact):.2%}")
    print(f"Tiempo de ejecucion puro: {end_time - start_time:.2f} s")

if __name__ == "__main__":
    run_vit_benchmark()