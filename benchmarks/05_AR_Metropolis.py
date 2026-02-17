import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import netket as nk
import optax
import time
from physics.hamiltonian import get_Hamiltonian
from models.vit import ARSpinViT_Manual

def run_ar_metropolis():
    print(">>> BENCHMARK 05: ARNN SOLO ANSATZ (Metropolis)")
    print(">>> Objetivo: Demostrar que sin Sampleo Directo, es lento.")
    print("---------------------------------------------------------")
    
    # 1. Sistema
    N = 10
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=1.0, alpha=3.0, hilbert=hi)

    # 2. Modelo (El Autoregresivo)
    # Nota: En NetKet, para usar ARNN como modelo normal, lo envolvemos
    model = ARSpinViT_Manual(hilbert=hi, embedding_dim=8, n_blocks=2)

    # 3. Sampler (EL LENTO)
    # Usamos MetropolisLocal. Esto ignora la capacidad AR del modelo.
    sampler = nk.sampler.MetropolisLocal(hi)

    # 4. Estado Variacional
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)

    # 5. Optimizador
    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

    # 6. Ejecutar
    start_time = time.time()
    log = nk.logging.JsonLog("resultado_benchmark_05", save_params=False)
    gs.run(n_iter=300, out=log, show_progress=True)
    end_time = time.time()
    
    print(f"\nResultados AR + Metropolis:")
    print(f"Energía: {log['Energy'].Mean[-1]:.6f}")
    print(f"Tiempo : {end_time - start_time:.2f} s")

if __name__ == "__main__":
    run_ar_metropolis()