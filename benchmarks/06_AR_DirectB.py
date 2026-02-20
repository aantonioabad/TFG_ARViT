import os
import sys

# Header
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import netket as nk
import optax
import time
from physics.hamiltonian import get_Hamiltonian
from models.vitB import ARSpinViT_Causal

def run_ar_direct():
    print(">>> BENCHMARK 06: MI ViT CAUSAL + SAMPLEO DIRECTO")
    print("---------------------------------------------------------")
    
    N = 10
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=1.0, alpha=3.0, hilbert=hi)

    
    model = ARSpinViT_Causal(
        hilbert=hi, 
        embedding_d=8,     
        n_blocks=2, 
        n_heads=2,         
        n_ffn_layers=1    
    )

    sampler = nk.sampler.ARDirectSampler(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
    vstate.chunk_size = 128
    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

    start_time = time.time()
    log = nk.logging.JsonLog("resultado_benchmark_06", save_params=False)
    gs.run(n_iter=1500, out=log, show_progress=True)
    end_time = time.time()
    
    print(f"\nEnergía final: {log['Energy'].Mean[-1]:.6f}")

if __name__ == "__main__":
    run_ar_direct()