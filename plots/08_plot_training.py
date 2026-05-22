import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_benchmark_training(log_path, benchmark_name, output_filename, exact_energy):
    print(f"Generando gráfica técnica para: {benchmark_name}...")
    
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
        'axes.titlesize': 11,   # Título pequeño
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.spines.top': True,
        'axes.spines.right': True,
    }):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        # Colores Pastel Técnicos
        color_data = "#5499C7"    # Azul acero pastel
        color_exact = "#F1948A"   # Salmón/Rojo pastel

        # 1. Datos crudos (única línea de datos)
        ax.plot(iters, energy_mean, color=color_data, linewidth=1.2, label="Energía VMC")
        
        # 2. Línea exacta
        ax.axhline(exact_energy, color=color_exact, linestyle="--", linewidth=1.8, 
                    label=f"Energía Exacta ({exact_energy:.4f})")

        # Título en MAYÚSCULAS
        ax.set_title(benchmark_name.upper(), pad=12)
        
        # Ejes
        ax.set_xlabel("Épocas")
        ax.set_ylabel(r"Parámetro de control, $H$")
        
        # Cuadrícula continua gris suave
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        
        # Resolución del eje Y
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))

        # Leyenda minimalista
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [ÉXITO] Gráfica guardada como '{output_filename}'\n")

if __name__ == "__main__":
    print("\n--- GENERANDO GRÁFICAS DE ENTRENAMIENTO (DATOS CRUDOS) ---\n")
    
    directorio_logs = "/content/drive/MyDrive/TFG_ARViT/graficas y resultados modelos/"
    
    # Energía exacta proporcionada
    E_EXACTA = -12.32525024471575
    
    logs_a_procesar = {
        "resultado_benchmark_02_Jastrow.log": "02 - Jastrow (Mean Field) + Metropolis",
        "resultado_benchmark_03_LSTM.log": "03 - LSTM + Metropolis",
        "resultado_benchmark_04_ViT.log": "04 - ViT Estándar + Metropolis",
        "resultado_benchmark_05_AR.log": "05 - ARNNDense + Metropolis",
        "resultado_benchmark_06_ARNN.log": "06 - ARNNDense + Direct Sampling",
        "resultado_benchmark_06_ARViT.log": "06 - ARViT + Direct Sampling",
    }

    for log_file, title in logs_a_procesar.items():
        ruta_completa = directorio_logs + log_file
        nombre_base = log_file.replace(".log", "")
        # Nuevo nombre para diferenciarlas
        png_name = directorio_logs + f"training_{nombre_base}_Final.png" 
        
        plot_benchmark_training(ruta_completa, title, png_name, exact_energy=E_EXACTA)