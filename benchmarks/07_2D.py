import os
import sys
import json
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import optax
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sps

# Configuramos el path para que encuentre tus carpetas models y physics
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from models.vitB import ARSpinViT_Causal
from physics.utils import BestIterKeeper

# Función constructora del Hamiltoniano (Usando Numpy estricto en CPU)
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

# Lector robusto del JSON con autoreparación
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
    # --- 1. CONFIGURACIÓN DEL BENCHMARK ---
    Lx, Ly = 4, 4
    N = Lx * Ly
    J_val = 1.0       # <- Asegúrate de que coincida con tus valores de test
    alpha_val = 2.0   # <- Asegúrate de que coincida con tus valores de test
    
    print(f"\n=========================================================")
    print(f"🚀 BENCHMARK 2D: ARViT + DIRECT SAMPLING (Malla {Lx}x{Ly})")
    print(f"=========================================================")
    
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian_2D(Lx=Lx, Ly=Ly, J=J_val, alpha=alpha_val, hilbert=hi)

    model = ARSpinViT_Causal(
        hilbert=hi,
        embedding_d=8,
        n_heads=2,
        n_blocks=2,
        n_ffn_layers=1
    )

    sampler = nk.sampler.ARDirectSampler(hi)
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
    
    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)
    keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

    # Configurar el logger
    log_base = os.path.join(current_dir, "resultado_benchmark_2D_ARViT")
    logger = nk.logging.JsonLog(log_base, mode="write")

    # --- 2. ENTRENAMIENTO ---
    print("Iniciando entrenamiento VMC (500 iteraciones)...")
    gs.run(n_iter=500, out=logger, show_progress=True, callback=keeper.update)
    
    # --- 3. RESTAURACIÓN DE LA MEJOR ÉPOCA ---
    print(f"\n[+] Entrenamiento terminado. Restaurando la mejor iteración...")
    vstate.parameters = keeper.best_state.parameters

    # --- 4. CÁLCULO DE MÉTRICAS FINALES (Solo para la mejor iteración) ---
    print("[+] Diagonalizando matriz exacta (ED) para N=16. Por favor, espera...")
    sp_h = H.to_sparse()
    eigvals, eigvecs = sps.eigsh(sp_h, k=1, which="SA")
    exact_energy = eigvals[0]
    psi_exact = eigvecs[:, 0]

    # Recalcular energía de forma limpia
    vmc_energy_best = vstate.expect(H).mean.real
    
    # Error relativo
    rel_error = abs((vmc_energy_best - exact_energy) / exact_energy) * 100
    
    # Fidelidad
    psi_vmc = vstate.to_array()
    psi_vmc = psi_vmc / jnp.linalg.norm(psi_vmc)
    psi_exact = psi_exact / jnp.linalg.norm(psi_exact)
    fidelity = abs(jnp.vdot(psi_vmc, psi_exact))**2

    print("\n=========================================================")
    print("🎯 RESULTADOS FINALES 2D (CALCULADOS SOBRE LA MEJOR ÉPOCA)")
    print("=========================================================")
    print(f"Energía Calculada (VMC) : {vmc_energy_best:.6f}")
    print(f"Energía Exacta (ED)     : {exact_energy:.6f}")
    print(f"Error Relativo          : {rel_error:.4f} %")
    print(f"Fidelidad Cuántica      : {fidelity:.6f}")
    print("=========================================================\n")
    
    # --- 5. GENERACIÓN DE GRÁFICAS (HASTA ITERACIÓN 300) ---
    print("[*] Generando gráficas profesionales de entrenamiento...")
    log_file_path = log_base + ".log"
    energies = extraer_energias_log(log_file_path)
    
    if energies:
        # Recortar hasta la iteración 300 (o el máximo disponible si fallase)
        limit = min(len(energies), 300)
        energies_300 = energies[:limit]
        iters_300 = np.arange(limit)
        
        # Calcular el error absoluto época a época
        abs_errors_300 = np.abs(np.array(energies_300) - exact_energy)

        # Gráfica 1: Convergencia de Energía
        plt.figure(figsize=(10, 6))
        plt.plot(iters_300, energies_300, label="Energía VMC", color='#1f77b4', linewidth=2)
        plt.axhline(y=exact_energy, color='#d62728', linestyle='--', linewidth=2, label=f"Energía Exacta ({exact_energy:.4f})")
        plt.xlabel("Épocas (Iteraciones)", fontsize=13, fontweight='bold')
        plt.ylabel("Energía $E$", fontsize=13, fontweight='bold')
        plt.title(f"Convergencia de la Energía - Malla 2D 4x4", fontsize=15, pad=15)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=12)
        plt.xlim(0, 300)
        plt.tight_layout()
        path_conv = os.path.join(current_dir, "convergencia_energia_2D.png")
        plt.savefig(path_conv, dpi=300)
        plt.close()

        # Gráfica 2: Error Absoluto Logarítmico
        plt.figure(figsize=(10, 6))
        plt.plot(iters_300, abs_errors_300, label=r"Error Absoluto $|E_{VMC} - E_{Exact}|$", color='#ff7f0e', linewidth=2)
        plt.yscale('log') # Escala logarítmica clave para ver la caída del error
        plt.xlabel("Épocas (Iteraciones)", fontsize=13, fontweight='bold')
        plt.ylabel("Error Absoluto (Escala Log)", fontsize=13, fontweight='bold')
        plt.title(f"Evolución del Error Absoluto - Malla 2D 4x4", fontsize=15, pad=15)
        plt.grid(True, which="both", linestyle=':', alpha=0.5)
        plt.legend(fontsize=12)
        plt.xlim(0, 300)
        plt.tight_layout()
        path_err = os.path.join(current_dir, "error_absoluto_2D.png")
        plt.savefig(path_err, dpi=300)
        plt.close()

        print(f"[√] Gráfica guardada: {path_conv}")
        print(f"[√] Gráfica guardada: {path_err}")
        
        # Copia automática a Google Drive
        drive_plots = "/content/drive/MyDrive/TFG_ARViT/plots/"
        if os.path.exists("/content/drive/MyDrive/"):
            os.system(f"mkdir -p {drive_plots}")
            os.system(f"cp {path_conv} {drive_plots}")
            os.system(f"cp {path_err} {drive_plots}")
            print("[√] Gráficas respaldadas en tu Google Drive.")
    else:
        print("[X] Hubo un problema al leer el archivo log. No se generaron las gráficas.")

if __name__ == "__main__":
    run_arvit_direct_2d()