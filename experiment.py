import os
# Configuración crítica para evitar conflictos de memoria con JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import netket as nk
import optax
import jax.numpy as jnp

# --- IMPORTACIONES DE NUESTROS MÓDULOS ---
from physics.hamiltonian import get_Hamiltonian
from models.vit import ARSpinViT

def run_experiment():
    print("===========================================")
    print(">>> INICIANDO AR-ViT (Estructura Modular)")
    print("===========================================")
    
    # --- 1. HIPERPARÁMETROS FÍSICOS ---
    N = 10          # Número de espines
    J = 1.0         # Interacción
    alpha = 3.0     # Alcance de interacción
    
    # Espacio de Hilbert (Espines 1/2)
    hi = nk.hilbert.Spin(s=0.5, N=N)
    
    # Obtener Hamiltoniano del módulo physics
    H = get_Hamiltonian(N, J, alpha, hi)

    # Cálculo Exacto (para referencia)
    print(">>> Calculando energía exacta (Lanczos)...")
    E_exact = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)[0]
    print(f"Energía Exacta: {E_exact:.6f}")

    # --- 2. MODELO (ARQUITECTURA) ---
    # [MODIFICADO]: Usamos la configuración "Lite" para estabilidad numérica.
    # ORIGINAL: embedding_d=16, n_heads=4 (Demasiado grande para N=10 con pocos samples)
    print(">>> Construyendo modelo ARSpinViT Lite...")
    model = ARSpinViT(
        hilbert=hi,
        embedding_d=8,    # Reducido para evitar overfitting
        n_heads=2,
        n_blocks=2, 
        n_ffn_layers=1 
    )
    
    # Sampler Autoregresivo (Exacto y eficiente)
    sampler = nk.sampler.ARDirectSampler(hi)

    # --- 3. ESTADO VARIACIONAL (VMC) ---
    # [MODIFICADO]: Aumentamos n_samples para garantizar N_samples > N_params
    # ORIGINAL: 1024 muestras.
    # CAMBIO: 2048 muestras.
    vstate = nk.vqs.MCState(
        sampler, 
        model, 
        n_samples=2048,  
        n_discard_per_chain=0,
        seed=42
    )

    print(f"Número de parámetros del modelo: {vstate.n_parameters}")
    
    # --- 4. OPTIMIZADOR ---
    # [MODIFICADO]: Learning Rate reducido y Diag Shift aumentado.
    # ORIGINAL: lr=0.01, diag_shift=0.01 (Causaba NaN/Explosión).
    # CAMBIO: lr=0.001 (Más lento pero seguro), diag_shift=0.1 (Freno inicial).
    optimizer = nk.optimizer.Sgd(learning_rate=0.001)
    sr = nk.optimizer.SR(diag_shift=0.1)

    # Driver de entrenamiento
    gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=sr)

    # --- 5. ENTRENAMIENTO ---
    print(">>> Entrenando (300 iteraciones)...")
    log = nk.logging.RuntimeLog()
    
    gs.run(n_iter=300, out=log, show_progress=True)

    # Resultados Finales
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