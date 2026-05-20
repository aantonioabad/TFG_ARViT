import os
import sys
import json
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from physics.hamiltonian import get_Hamiltonian

def plot_metropolis_training(log_path, phase_name, output_filename, exact_energy, alpha, J):
    if not os.path.exists(log_path):
        print(f"  [ERROR] No se encuentra el log en: {log_path}")
        return

    with open(log_path, 'r') as f:
        data = json.load(f)
        
    iters = data['Energy']['iters']
    energy_mean = [e.get('real', 0.0) if isinstance(e, dict) else float(np.real(e)) for e in data['Energy']['Mean']]

    # 1. Encontrar mejor iteración
    best_idx = np.argmin(energy_mean)
    best_energy = energy_mean[best_idx]
    best_iter = iters[best_idx]
    
    # 2. Calcular error relativo
    error_relativo = abs(best_energy - exact_energy) / abs(exact_energy)
    
    # 3. LEER LA FIDELIDAD INYECTADA
    fidelidad_val = data.get('Best_Fidelity', None)
    fidelidad_str = f"{fidelidad_val:.6f}" if fidelidad_val is not None else "N/A"

    # Generar la gráfica
    with plt.rc_context({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': True, 'axes.spines.right': True}):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        
        ax.plot(iters, energy_mean, color="#2E86C1", linewidth=1.2, label=f"ARViT (Metropolis) | $\\alpha={alpha}$")
        ax.axhline(exact_energy, color="#E74C3C", linestyle="--", linewidth=1.8, label=f"Energía Exacta ({exact_energy:.4f})")

        # Texto formateado para la leyenda
        datos_leyenda = (
            f"--- Mejor Época ({best_iter}) ---\n"
            f"Mejor $\\langle H \\rangle$: {best_energy:.4f}\n"
            f"Err. Relativo: {error_relativo:.2e}\n"
            f"Fidelidad $F$: {fidelidad_str}"
        )
        ax.plot([], [], ' ', label=datos_leyenda)

        ax.set_title(f"Metropolis MCMC: {phase_name} (J={J}, $\\alpha$={alpha})", pad=12)
        ax.set_xlabel("Épocas")
        ax.set_ylabel(r"Energía, $\langle H \rangle$")
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
        ax.legend(loc="upper right", frameon=True, fontsize=9, facecolor='#FDFEFE', edgecolor='#BDC3C7')
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [√] Gráfica guardada: {os.path.basename(output_filename)}")

if __name__ == "__main__":
    import netket as nk
    N = 10
    drive_dir = "/content/drive/MyDrive/TFG_ARViT/Fase_ J_alpha/metropolis/"
    
    experimentos = [
        (6.0, 7.0, "Fase AFM"), (6.0, -4.0, "Fase FM"),
        (2.5, -4.0, "Fase FM"), (2.5, 2.75, "Región AFM / Crítica"),
        (1.0, -2.0, "Fase FM"), (1.0, 7.0, "Fase AFM / Frustrada")
    ]

    for alpha, J, fase in experimentos:
        hi = nk.hilbert.Spin(s=1/2, N=N)
        H = get_Hamiltonian(N=N, J=J, alpha=alpha, hilbert=hi)
        H_sparse = H.to_sparse()
        evals, _ = scipy.sparse.linalg.eigsh(H_sparse, k=1, which="SA")
        E_exacta = float(evals[0])

        ruta_log = os.path.join(drive_dir, f"resultado_metropolis_alpha{alpha}_J{J}.log")
        nombre_grafica = os.path.join(drive_dir, f"training_metropolis_{J}_{alpha}_.png")
        
        plot_metropolis_training(ruta_log, fase, nombre_grafica, exact_energy=E_exacta, alpha=alpha, J=J)