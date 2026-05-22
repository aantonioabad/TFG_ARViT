import os
import sys
import json
import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Ajuste de rutas para importar tu Hamiltoniano
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from physics.hamiltonian import get_Hamiltonian

def plot_direct_training(log_path, phase_name, output_filename, exact_energy, alpha, J):
    if not os.path.exists(log_path):
        print(f"  [ERROR] No se encuentra el log en: {log_path}")
        return

    with open(log_path, 'r') as f:
        data = json.load(f)
        
    iters = data['Energy']['iters']
    energy_mean = [e.get('real', 0.0) if isinstance(e, dict) else float(np.real(e)) for e in data['Energy']['Mean']]

    # --- 1. EXTRACCIÓN DE MÉTRICAS PARA LA LEYENDA ---
    best_idx = int(np.argmin(energy_mean))
    best_energy = energy_mean[best_idx]
    best_iter = iters[best_idx]
    
    error_relativo = abs(best_energy - exact_energy) / abs(exact_energy)
    
    # Leer la fidelidad inyectada en el script anterior
    fidelidad_val = data.get('Best_Fidelity', None)
    fidelidad_str = f"{fidelidad_val:.6f}" if fidelidad_val is not None else "N/A"

    # --- 2. GENERACIÓN DE LA GRÁFICA ---
    with plt.rc_context({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': True, 'axes.spines.right': True}):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        
        # Línea de convergencia (usamos un verde o morado distinto para diferenciar del azul de Metropolis si quieres)
        ax.plot(iters, energy_mean, color="#27AE60", linewidth=1.2, label=f"ARViT (Directo) | $\\alpha={alpha}$")
        ax.axhline(exact_energy, color="#E74C3C", linestyle="--", linewidth=1.8, label=f"Energía Exacta ({exact_energy:.4f})")

        # Texto formateado exactamente como en tu imagen
        datos_leyenda = (
            f"--- Mejor Época ({best_iter}) ---\n"
            f"Mejor $\\langle H \\rangle$: {best_energy:.4f}\n"
            f"Err. Relativo: {error_relativo:.2e}\n"
            f"Fidelidad $F$: {fidelidad_str}"
        )
        ax.plot([], [], ' ', label=datos_leyenda)

        ax.set_title(f"Muestreo Directo: {phase_name} (J={J}, $\\alpha$={alpha})", pad=12)
        ax.set_xlabel("Épocas")
        ax.set_ylabel(r"Energía, $\langle H \rangle$")
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
        
        # Cuadro de leyenda formal
        ax.legend(loc="upper right", frameon=True, fontsize=9, facecolor='#FDFEFE', edgecolor='#BDC3C7')
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [ÉXITO] Guardada gráfica: {os.path.basename(output_filename)}")


if __name__ == "__main__":
    import netket as nk
    N = 10  # Número de espines
    drive_dir = "/content/drive/MyDrive/TFG_ARViT/Fase_ J_alpha/"
    
    # Lista exacta de los 6 experimentos solicitados
    experimentos = [
        (6.0, 7.0, "Fase AFM"),
        (6.0, -4.0, "Fase FM"),
        (2.5, -4.0, "Fase FM"),
        (2.5, 2.75, "Región AFM / Crítica"),
        (1.0, -2.0, "Fase FM"),
        (1.0, 7.0, "Fase AFM / Frustrada")
    ]

    print("\n" + "="*70)
    print("📊 PLOTEANDO 6 PUNTOS CON MUESTREO DIRECTO (CON MÉTRICAS) 📊")
    print("="*70 + "\n")

    for alpha, J, fase in experimentos:
        print(f"\n>>> Procesando J={J} | alpha={alpha} <<<")
        
        # 1. Calcular energía exacta
        hi = nk.hilbert.Spin(s=1/2, N=N)
        H = get_Hamiltonian(N=N, J=J, alpha=alpha, hilbert=hi)
        H_sparse = H.to_sparse()
        evals, _ = scipy.sparse.linalg.eigsh(H_sparse, k=1, which="SA")
        E_exacta = float(evals[0])
        print(f"  [+] Energía Exacta de referencia: {E_exacta:.6f}")

        # 2. Ruta del log (asegurándonos de leer el 'resultado_direct_...' que generaste)
        ruta_log = os.path.join(drive_dir, f"resultado_direct_alpha{alpha}_J{J}.log")
        
        # 3. Nombre de la imagen de salida
        nombre_grafica = os.path.join(drive_dir, f"training_direct_{J}_{alpha}_.png")
        
        # 4. Generar gráfica
        plot_direct_training(ruta_log, fase, nombre_grafica, exact_energy=E_exacta, alpha=alpha, J=J)

    print("\n[√] ¡Misión cumplida! Tienes tus 6 gráficas de Muestreo Directo en Drive.")