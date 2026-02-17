import os
import sys

# --- 1. HEADER (Para encontrar las carpetas) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["JAX_PLATFORM_NAME"] = "cpu"

# --- 2. IMPORTS ---
import netket as nk
import optax
import time
from physics.hamiltonian import get_Hamiltonian


from models.vit_standard import BatchedSpinViT

def run_vit_benchmark():
    print(">>> BENCHMARK 04: ViT ESTÁNDAR (Paper Reference)")
    print(">>> Método: Ansatz No-Autoregresivo + Metropolis Sampling")
    print("---------------------------------------------------------")
    
    # 1. Sistema Físico (Tu Hamiltoniano de largo alcance)
    N = 10
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=1.0, alpha=3.0, hilbert=hi)

    # Cargar energía exacta para comparar
    try:
        with open("benchmark_exact.txt", "r") as f:
            E_exact = float(f.read())
    except:
        E_exact = None

    # 2. El Modelo (El del Paper)
   
    model = BatchedSpinViT(
        token_size=1,
        embedding_d=8,
        n_heads=2,
        n_blocks=2,
        n_ffn_layers=1,
        final_architecture=[8, 4], 
        is_complex=False
    )

    
    sampler = nk.sampler.MetropolisLocal(hi)

    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)

    # 5. Optimizador
    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

    # 6. Entrenamiento
    print("Entrenando ViT (Metropolis)...")
    start_time = time.time()
    
    log = nk.logging.JsonLog("resultado_benchmark_04", save_params=False)
    gs.run(n_iter=300, out=log, show_progress=True)
    
    end_time = time.time()
    E_final = log["Energy"].Mean[-1]
    
    # 7. Resultados
    print(f"\n>>> RESULTADOS 04_ViT:")
    print(f"Energía ViT : {E_final:.6f}")
    if E_exact:
        print(f"Energía Exacta: {E_exact:.6f}")
        print(f"Error       : {abs((E_final - E_exact)/E_exact):.2%}")
    print(f"Tiempo Total: {end_time - start_time:.2f} s")

if __name__ == "__main__":
    run_vit_benchmark()