import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_long_range_training(log_path, phase_name, output_filename, exact_energy):
    print(f"Generando gráfica para: {phase_name}...")
    
    if not os.path.exists(log_path):
        print(f"  [ERROR] No se encuentra el archivo {log_path}.")
        return

    with open(log_path, 'r') as f:
        data = json.load(f)
        
    iters = data['Energy']['iters']
    energy_mean = []
    for e in data['Energy']['Mean']:
        if isinstance(e, dict):
            energy_mean.append(e.get('real', 0.0))
        else:
            energy_mean.append(float(np.real(e)))

    # ESTÉTICA DE ARTÍCULO CIENTÍFICO (LaTeX-like)
    with plt.rc_context({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.spines.top': True,
        'axes.spines.right': True,
    }):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        # Colores: Usaremos un tono morado/índigo pastel muy elegante para diferenciarlo de los benchmarks
        color_data = "#7D3C98"    # Púrpura pastel oscuro
        color_exact = "#F1948A"   # Salmón pastel

        # Datos crudos
        ax.plot(iters, energy_mean, color=color_data, linewidth=1.2, label="Energía VMC (ARViT)")
        
        # Línea exacta
        ax.axhline(exact_energy, color=color_exact, linestyle="--", linewidth=1.8, 
                    label=f"Energía Exacta ({exact_energy:.4f})")

        ax.set_title(phase_name.upper(), pad=12)
        ax.set_xlabel("Épocas")
        ax.set_ylabel(r"Valor Esperado de la Energía, $\langle H \rangle$")
        
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
        ax.legend(loc="upper right", frameon=False, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [ÉXITO] Gráfica guardada como '{output_filename}'\n")

if __name__ == "__main__":
    print("\n--- GENERANDO GRÁFICAS DEL DIAGRAMA DE FASES (LONG-RANGE) ---\n")
    
    # 1. RUTA: Asegúrate de que apunta a donde se guardan los logs
    directorio_logs = "/content/drive/MyDrive/TFG_ARViT/Fase_J_alpha/"
    

    exact_energies = {
        -4.0:  -51.733292,  
        -2.0: -26.603024,   
         1.0: -12.204841,   
         4.75: -41.702796,  
         7.0: -60.968720    
    }
    
    # Mapeo de los valores de J a su nombre físico (para el título de la gráfica)
    experimentos = {
        -4.0: "J = -4.0 | Fase FM Profunda",
        -2.0: "J = -2.0 | Transición Crítica FM",
         1.0: "J =  1.0 | Fase Paramagnética (Desorden)",
         4.75: "J =  4.75 | Transición Crítica AFM",
         7.0: "J =  7.0 | Fase AFM Profunda"
    }

    for J, titulo in experimentos.items():
        nombre_log = f"resultado_LR_alpha2.5_J{J}.log"
        ruta_completa = os.path.join(directorio_logs, nombre_log)
        
        png_name = os.path.join(directorio_logs, f"training_LR_J{J}_Paper.png")
        
        # Obtenemos la energía exacta de nuestro diccionario
        E_exacta = exact_energies[J]
        
        plot_long_range_training(ruta_completa, titulo, png_name, exact_energy=E_exacta)