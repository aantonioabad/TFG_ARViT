import os
import sys
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import optax
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sps

base_tfg_dir = "/content/TFG_ARViT"
if base_tfg_dir not in sys.path:
    sys.path.append(base_tfg_dir)

from models.vitB import ARSpinViT_Causal
from physics.utils import BestIterKeeper


def get_Hamiltonian_2D(Lx: int, Ly: int, J: float, alpha: float, h: float = 1.0, hilbert=None):
    N = Lx * Ly
    if hilbert is None:
        hilbert = nk.hilbert.Spin(s=0.5, N=N)
    
    graph = nk.graph.Grid(extent=[Lx, Ly], pbc=True)
    distances = graph.distances()
    H = nk.operator.LocalOperator(hilbert)
    
    sigmax = np.array([[0, 1], [1, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])
    
    for i in range(N):
        H += nk.operator.LocalOperator(hilbert, -h * sigmax, [i])
        
    for i in range(N):
        for j in range(i + 1, N):
            dist = distances[i][j]
            if dist > 0:
                coupling = J / (dist ** alpha)
                term = coupling * np.kron(sigmaz, sigmaz)
                H += nk.operator.LocalOperator(hilbert, term, [i, j])
                
    return H

def extraer_energias_log(log_path):
    if not os.path.exists(log_path):
        return []
    with open(log_path, 'r') as f:
        text = f.read().strip()
        
    decoder = json.JSONDecoder()
    data_list = []
    idx = 0
    while idx < len(text):
        text_substr = text[idx:].lstrip()
        if not text_substr: break
        try:
            obj, next_idx = decoder.raw_decode(text_substr)
            data_list.append(obj)
            idx += next_idx
        except json.JSONDecodeError:
            break
            
    if not data_list: return []
    data = data_list[-1]
    energy_dict = data.get("Energy", {})
    e_mean_list = energy_dict.get("Mean", energy_dict.get("mean", energy_dict.get("value", [])))
    
    energies = [e.get("real", e.get("Mean", e.get("mean", 0.0))) if isinstance(e, dict) else (e.real if isinstance(e, complex) else float(e)) for e in e_mean_list]
    return energies

def run_arvit_direct_2d():
    
    Lx, Ly = 2, 2
    N = Lx * Ly
    J_val = 1.0
    alpha_val = 2.0
    
    print(f"\n=========================================================")
    print(f"🚀 BENCHMARK 2D: ARViT + DIRECT (Malla {Lx}x{Ly})")
    print(f"=========================================================")
    
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian_2D(Lx=Lx, Ly=Ly, J=J_val, alpha=alpha_val, hilbert=hi)

    model = ARSpinViT_Causal(
        hilbert=hi, embedding_d=8, n_heads=2, n_blocks=2, n_ffn_layers=1
    )

    sampler = nk.sampler.ARDirectSampler(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
    
    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)
    keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

    
    log_base = os.path.join(base_tfg_dir, "resultado_benchmark_2D_ARViT")
    logger = nk.logging.JsonLog(log_base, mode="write")

    
    print("Iniciando entrenamiento VMC (1000 iteraciones)...")
    start_time = time.time()
    gs.run(n_iter=1000, out=logger, show_progress=True, callback=keeper.update)
    exec_time = time.time() - start_time
    
    
    vstate.parameters = keeper.best_state.parameters
    print("[+] Diagonalizando matriz exacta (ED)...")
    sp_h = H.to_sparse()
    eigvals, eigvecs = sps.eigsh(sp_h, k=1, which="SA")
    exact_energy = eigvals[0]
    psi_exact = eigvecs[:, 0]

   
    vmc_energy_best = vstate.expect(H).mean.real
    
    rel_error = abs((vmc_energy_best - exact_energy) / exact_energy) * 100
    
   
    psi_vmc = vstate.to_array()
    psi_vmc_norm = psi_vmc / jnp.linalg.norm(psi_vmc)
    psi_exact_norm = psi_exact / jnp.linalg.norm(psi_exact)
    fidelity = abs(jnp.vdot(psi_vmc_norm, psi_exact_norm))**2

   
    prob_vmc = np.abs(psi_vmc_norm)**2
    prob_exact = np.abs(psi_exact_norm)**2
    pearson_corr = np.corrcoef(prob_vmc, prob_exact)[0, 1]
    pearson_dev = 1.0 - pearson_corr

    print("\n=========================================================")
    print(" RESULTADOS FINALES 2D (MEJOR ÉPOCA RESTAURADA)")
    print("=========================================================")
    print(f"Energía Calculada (VMC) : {vmc_energy_best:.6f}")
    print(f"Energía Exacta (ED)     : {exact_energy:.6f}")
    print(f"Error Relativo          : {rel_error:.4f} %")
    print(f"Fidelidad Cuántica      : {fidelity:.6f}")
    print(f"Desviación Pearson      : {pearson_dev:.6f}")
    print(f"Tiempo de Ejecución     : {exec_time:.2f} segundos")
    print("=========================================================\n")
    
    
    print("[*] Generando gráficas profesionales de entrenamiento...")
    energies = extraer_energias_log(log_base + ".log")
    
    if energies:
        all_iters = np.arange(len(energies))
        abs_errors_all = np.abs(np.array(energies) - exact_energy)

       
        plt.figure(figsize=(10, 6))
        
       
        leyenda_vmc = (f"Energía VMC\n"
                       f"Métricas Finales (Best Iter):\n"
                       f"• Error Rel: {rel_error:.4f}%\n"
                       f"• Fidelidad: {fidelity:.4f}")
        
        plt.plot(all_iters, energies, label=leyenda_vmc, color='#1f77b4', linewidth=2)
        plt.axhline(y=exact_energy, color='#d62728', linestyle='--', linewidth=2, label=f"Energía Exacta ({exact_energy:.4f})")
        
        plt.xlabel("Épocas (Iteraciones)", fontsize=13, fontweight='bold')
        plt.ylabel("Energía $E$", fontsize=13, fontweight='bold')
        plt.title(f"Convergencia de la Energía - Malla 2D {Lx}x{Ly}", fontsize=15, pad=15)
        plt.grid(True, linestyle=':', alpha=0.7)
        
       
        plt.legend(fontsize=10, loc='upper right') 
        
        
        plt.xlim(0, 450) 
        plt.tight_layout()
        
        
        path_conv = os.path.join(base_tfg_dir, "convergencia_energia_2D.png")
        plt.savefig(path_conv, dpi=300)
        plt.close()

        
        plt.figure(figsize=(10, 6))
        plt.plot(all_iters, abs_errors_all, label=r"Error Absoluto $|E_{VMC} - E_{Exact}|$", color='#ff7f0e', linewidth=2)
        plt.yscale('log')
        plt.xlabel("Épocas (Iteraciones)", fontsize=13, fontweight='bold')
        plt.ylabel("Error Absoluto (Escala Log)", fontsize=13, fontweight='bold')
        plt.title(f"Evolución del Error Absoluto - Malla 2D {Lx}x{Ly}", fontsize=15, pad=15)
        plt.grid(True, which="both", linestyle=':', alpha=0.5)
        plt.legend(fontsize=12)
        
       
        plt.xlim(0, 1000) 
        plt.tight_layout()
        
       
        path_err = os.path.join(base_tfg_dir, "error_relativo_2D.png")
        plt.savefig(path_err, dpi=300)
        plt.close()

        print(f"[√] Gráfica guardada en TFG_ARViT: {path_conv}")
        print(f"[√] Gráfica guardada en TFG_ARViT: {path_err}")
        
    else:
        print("[X] Problema leyendo el log. No se generaron gráficas.")

if __name__ == "__main__":
    run_arvit_direct_2d()