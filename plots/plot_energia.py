import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def extraer_energias(log_path):
    """Extrae las energías iteración por iteración esquivando errores de formato."""
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

def plot_benchmark_training(log_path, benchmark_name, output_filename, exact_energy):
    print(f"Generando gráfica técnica para: {benchmark_name}...")
    
    energy_mean = extraer_energias(log_path)
    
    if not energy_mean:
        print(f"  [ERROR] No se pudo leer {log_path} o el archivo está vacío.")
        return

    iters = range(len(energy_mean))

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
        ax.set_xlabel("Épocas (Iteraciones)")
        ax.set_ylabel(r"Energía $\langle H \rangle$") # Cambiado de Parámetro de control H a Energía <H>
        
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
    print("\n--- GENERANDO GRÁFICAS INDIVIDUALES DE ENTRENAMIENTO ---\n")
    
    # Directorio base (Local en Colab)
    directorio_base = "/content/TFG_ARViT/"
    
    # Energía exacta de tu modelo 1D N=10 J=1 alpha=2 (Asegúrate de que este es el valor que quieres)
    E_EXACTA = -12.32525024471575 
    
    # Lista de los logs que tienes en la captura
    logs_a_procesar = {
        "resultado_benchmark_02_Jastrow.log": "02 - Jastrow (Mean Field) + Metropolis",
        "resultado_benchmark_03_RBM.log": "03 - RBM + Metropolis",  # Corregido de LSTM a RBM
        "resultado_benchmark_05_AR.log": "05 - ARNNDense + Metropolis",
        "resultado_benchmark_06_ARNN.log": "06 - ARNNDense + Direct Sampling",
        "resultado_benchmark_06_ARViT.log": "06 - ARViT + Direct Sampling"
    }

    for log_file, title in logs_a_procesar.items():
        ruta_completa = os.path.join(directorio_base, log_file)
        nombre_base = log_file.replace(".log", "")
        
        # Guardar en la misma carpeta base
        png_name = os.path.join(directorio_base, f"training_{nombre_base}.png")
        
        plot_benchmark_training(ruta_completa, title, png_name, exact_energy=E_EXACTA)