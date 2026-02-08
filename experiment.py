import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import netket as nk
import optax  # Necesario para el clipping
import jax.numpy as jnp

from physics.hamiltonian import get_Hamiltonian
from models.vit import ARSpinViT

def run_experiment():
    print("===========================================")
    print(">>> INICIANDO AR-ViT (Modo Anti-Explosión)")
    print("===========================================")
    
    # --- HIPERPARÁMETROS ---
    N = 10
    J = 1.0
    alpha = 3.0
    
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J, alpha, hi)

    print(">>> Calculando energía exacta...")
    E_exact = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)[0]
    print(f"Energía Exacta: {E_exact:.6f}")

    # --- MODELO ---
    print(">>> Construyendo modelo...")
    model = ARSpinViT(
        hilbert=hi,
        embedding_d=8,
        n_heads=2,
        n_blocks=2, 
        n_ffn_layers=1 
    )
    
    sampler = nk.sampler.ARDirectSampler(hi)

    # --- MUESTRAS ---
    # Mantenemos alto para estabilidad
    vstate = nk.vqs.MCState(
        sampler, 
        model, 
        n_samples=2048,  
        n_discard_per_chain=0,
        seed=42
    )

    print(f"Parámetros: {vstate.n_parameters}")
    
    # --- [FIX CRÍTICO 2] OPTIMIZADOR CON CLIPPING ---
    # Usamos Adam (mejor para Transformers) en vez de SGD.
    # Clip Global Norm: Si el vector de gradiente es muy largo, lo recorta a 1.0
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # <--- Freno de emergencia
        optax.adam(learning_rate=0.001)  # <--- Adam suele ser más estable que SGD
    )
    
    # --- [FIX CRÍTICO 3] PRECONDICIONADOR SVD ---
    # Usamos solver='svd'. Es más lento pero matemáticamente indestructible frente a matrices singulares.
    sr = nk.optimizer.SR(
        diag_shift=0.1, 
        solver=nk.optimizer.solver.svd  # <--- SVD evita errores de inversión
    )

    gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=sr)

    print(">>> Entrenando (Con protecciones activadas)...")
    log = nk.logging.RuntimeLog()
    
    # Entrenamos un poco más para ver si converge
    gs.run(n_iter=300, out=log, show_progress=True)

    E_final = log["Energy"].Mean[-1]
    error = abs((E_final - E_exact) / E_exact)
    
    print("\n===========================================")
    print(">>> RESULTADOS FINALES")
    print(f"Energía Exacta : {E_exact:.6f}")
    print(f"Energía VMC    : {E_final:.6f}")
    print(f"Error Relativo : {error:.2%}")
    print("===========================================")

if __name__ == "__main__":
    run_experiment()