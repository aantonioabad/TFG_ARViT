import sys
current_dir = os.path.dirname(os.path.abspath(__file__))


parent_dir = os.path.dirname(current_dir)

sys.path.append(parent_dir)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"



from physics.hamiltonian import get_Hamiltonian
import os
import netket as nk
import optax
from models.vit import ARSpinViT_Manual

def run():
    print(">>> CARGANDO ARQUITECTURA...")
    
    # 1. Sistema Físico
    N = 10
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=1.0, alpha=3.0, hilbert=hi)
    
    E_exact = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)[0]
    print(f"Energía Exacta: {E_exact:.5f}")

    # 2. Modelo Modular
    model = ARSpinViT_Manual(
        hilbert=hi,
        embedding_d=8,
        n_heads=2,
        n_blocks=2,
        n_ffn_layers=1
    )

    # 3. Variational State
    sampler = nk.sampler.ARDirectSampler(hi)
    
    
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
    print(f"Parámetros: {vstate.n_parameters} | Muestras: {vstate.n_samples}")

    # 4. Optimizador
    
    op = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=0.001)
    )

    
    gs = nk.driver.VMC_SR(H, op, variational_state=vstate, diag_shift=0.1)
    
    log = nk.logging.RuntimeLog()
    gs.run(n_iter=300, out=log, show_progress=True)

    E_final = log["Energy"].Mean[-1]
    print(f"\n>>> FINAL: VMC={E_final:.5f} | Exacta={E_exact:.5f}")
    
   
    err = abs((E_final - E_exact)/E_exact)
    print(f"Error: {err:.2%}")

if __name__ == "__main__":
    run()