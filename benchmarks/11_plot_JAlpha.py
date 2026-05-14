import os
import sys
import json
import glob
import numpy as np
import netket as nk
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Ajuste de rutas para importar tu Hamiltoniano
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from physics.hamiltonian import get_Hamiltonian

def plot_training(log_path, phase_name, output_filename, exact_energy, alpha, J):
    if not os.path.exists(log_path):
        print(f"  [ERROR] No se encuentra el log en: {log_path}")
        return

    with open(log_path, 'r') as f:
        data = json.load(f)
        
    iters = data['Energy']['iters']
    energy_mean = [e.get('real', 0.0) if isinstance(e, dict) else float(np.real(e)) for e in data['Energy']['Mean']]

    with plt.rc_context({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': True, 'axes.spines.right': True}):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        
        # Gráfica con el valor de alpha en la leyenda, como pediste
        ax.plot(iters, energy_mean, color="#7D3C98", linewidth=1.2, label=f"VMC (ARViT) | $\\alpha={alpha}$")
        ax.axhline(exact_energy, color="#F1948A", linestyle="--", linewidth=1.8, label=f"Energía Exacta ({exact_energy:.4f})")

        ax.set_title(f"{phase_name} (J={J}, $\\alpha$={alpha})", pad=12)
        ax.set_xlabel("Épocas")
        ax.set_ylabel(r"Energía, $\langle H \rangle$")
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
        ax.legend(loc="upper right", frameon=False, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close() # Asegura que sean gráficas 100% separadas
        print(f"  [ÉXITO] Guardada: {output_filename}")


if __name__ == "__main__":
    N = 10  # Número de espines para calcular la energía exacta
    drive_dir = "/content/drive/MyDrive/TFG_ARViT/Fase_ J_alpha/"
    
    # Valores de J y alpha EXACTOS según tus capturas de pantalla
    experimentos = {
        2.5: {
            -4.0: "Fase FM", 
            -2.0: "Crit FM", 
             1.0: "Para", 
             4.75: "Crit AFM", 
             7.0: "Fase AFM"
        },
        6.0: {
            -4.0: "Fase FM", 
            -3.0: "Crit FM", 
             1.0: "Para", 
             3.0: "Crit AFM", 
             7.0: "Fase AFM"
        }
    }

    exact_energies_summary = {alpha: {} for alpha in experimentos.keys()}

    print("\n--- CALCULANDO ENERGÍAS Y GENERANDO 10 GRÁFICAS INDIVIDUALES ---\n")

    for alpha, js_dict in experimentos.items():
        for J, titulo in js_dict.items():
            print(f"\n>>> Procesando J={J} | alpha={alpha} <<<")
            
            # 1. Calcular la energía exacta sobre la marcha
            hi = nk.hilbert.Spin(s=1/2, N=N)
            H = get_Hamiltonian(N=N, J=J, alpha=alpha, hilbert=hi)
            H_sparse = H.to_sparse()
            evals, _ = scipy.sparse.linalg.eigsh(H_sparse, k=1, which="SA")
            E_exacta = float(evals[0])
            exact_energies_summary[alpha][J] = E_exacta
            print(f"  [+] Energía Exacta Calculada: {E_exacta:.6f}")

            # 2. Buscar el archivo de log correspondiente en Drive usando comodines
            patron_busqueda = os.path.join(drive_dir, f"resultado_LR_alpha{alpha}_J{J}*")
            archivos_encontrados = glob.glob(patron_busqueda)
            
            if not archivos_encontrados:
                print(f"  [OMITIDO] No hay log de entrenamiento guardado para este punto.")
                continue
                
            ruta_log = archivos_encontrados[0]
            
            # 3. Formato exacto de nombre de archivo solicitado: training_LR_J_alpha_.png
            png_name = os.path.join(drive_dir, f"training_LR_{J}_{alpha}_.png")
            
            # 4. Generar la gráfica individual
            plot_training(ruta_log, titulo, png_name, exact_energy=E_exacta, alpha=alpha, J=J)

    # 5. Imprimir el bloque de energías al final
    print("\n" + "="*50)
    print(" ENERGÍAS EXACTAS CALCULADAS")
    print("="*50)
    print("exact_energies = {")
    for alpha_val in exact_energies_summary:
        print(f"    {alpha_val}: {{")
        for j_val, e_val in exact_energies_summary[alpha_val].items():
            print(f"        {j_val}: {e_val:.6f},")
        print("    },")
    print("}")
    print("="*50 + "\n")