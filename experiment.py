import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import netket as nk
import optax
from physics.hamiltonian import get_Hamiltonian
# Importamos la nueva clase modular
from models.vit import ARSpinViT_Manual

def run():
    print(">>> CARGANDO ARQUITECTURA MODULAR...")
    
    # 1. Sistema Físico
    N = 10
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=1.0, alpha=3.0, hilbert=hi)
    
    E_exact = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)[0]
    print(f"Energía Exacta: {E_exact:.5f}")

   
    model = ARSpinViT_Manual(
        hilbert=hi,
        embedding_d=8,
        n_heads=2,
        n_blocks=2,
        n_ffn_layers=1
    )

    # 3. Variational State
    # Usamos sampler ARDirectSampler (Exacto para ARNN)
    sampler = nk.sampler.ARDirectSampler(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=1024, seed=42)

    # 4. Optimizador (Adam es mejor para Transformers que SGD)
    # Clip global norm evita explosiones
    op = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=0.005)
    )
    sr = nk.optimizer.SR(diag_shift=0.05) # SR suave

    # 5. Entrenar
    gs = nk.driver.VMC(H, op, variational_state=vstate, preconditioner=sr)
    log = nk.logging.RuntimeLog()
    gs.run(n_iter=300, out=log, show_progress=True)

    E_final = log["Energy"].Mean[-1]
    print(f"\n>>> FINAL: VMC={E_final:.5f} | Exacta={E_exact:.5f}")
    print(f"Error: {abs((E_final - E_exact)/E_exact):.2%}")

if __name__ == "__main__":
    run()