import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def extraer_datos_grafica(log_path):
    """Extrae las energías y busca la fidelidad si está guardada en el log."""
    if not os.path.exists(log_path):
        return [], None
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
            
    if not data_list: return [], None
    
    # Buscar energías en el último objeto válido
    data = data_list[-1]
    energy_dict = data.get("Energy", {})
    e_mean_list = energy_dict.get("Mean", energy_dict.get("mean", energy_dict.get("value", [])))
    energies = [e.get("real", e.get("Mean", e.get("mean", 0.0))) if isinstance(e, dict) else (e.real if isinstance(e, complex) else float(e)) for e in e_mean_list]
    
    # Rastrear la Fidelidad en todos los objetos JSON extraídos
    fidelidad = None
    for obj in data_list:
        if isinstance(obj, dict):
            if 'Best_Fidelity' in obj:
                fidelidad = obj['Best_Fidelity']
            elif 'Fidelity' in obj:
                fidelidad = obj['Fidelity']
                
    return energies, fidelidad

def plot_benchmark_training(log_path, benchmark_name, output_filename, exact_energy):
    print(f"Generando gráfica técnica para: {benchmark_name}...")
    
    energy_mean, fidelidad = extraer_datos_grafica(log_path)
    
    if not energy_mean:
        print(f"  [ERROR] No se pudo leer {log_path} o el archivo está vacío.")
        return

    iters = range(len(energy_mean))
    
    # --- CÁLCULO DE MÉTRICAS ---
    best_energy = min(energy_mean)
    rel_error = abs((best_energy - exact_energy) / exact_energy) * 100
    
    # Construir etiqueta de la leyenda con las métricas
    if fidelidad is not None:
        label_vmc = (f"Energía VMC\n"
                     f"Error Rel: {rel_error:.4f}%\n"
                     f"Fidelidad: {fidelidad:.4f}")
    else:
        label_vmc = (f"Energía VMC\n"
                     f"Error Rel: {rel_error:.4f}%\n"
                     f"Fidelidad: N/A")

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

        color_data = "#5499C7"    # Azul acero pastel
        color_exact = "#F1948A"   # Salmón/Rojo pastel

        # 1. Datos crudos con la nueva leyenda de métricas
        ax.plot(iters, energy_mean, color=color_data, linewidth=1.2, label=label_vmc)
        
        # 2. Línea exacta
        ax.axhline(exact_energy, color=color_exact, linestyle="--", linewidth=1.8, 
                    label=f"Energía Exacta ({exact_energy:.4f})")

        ax.set_title(benchmark_name.upper(), pad=12)
        ax.set_xlabel("Épocas (Iteraciones)")
        ax.set_ylabel(r"Energía $\langle H \rangle$")
        
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))

        # Leyenda ajustada para que quepa el bloque de texto
        ax.legend(loc="upper right", frameon=True, fontsize=9, edgecolor='black', facecolor='white', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [ÉXITO] Gráfica guardada como '{os.path.basename(output_filename)}'\n")

if __name__ == "__main__":
    print("\n--- GENERANDO GRÁFICAS INDIVIDUALES DE ENTRENAMIENTO ---\n")
    
    # Directorio base (Local en Colab)
    directorio_base = "/content/TFG_ARViT/"
    
    # Energía exacta de referencia
    E_EXACTA = -12.32525024471575 
    
    # Diccionario de logs
    logs_a_procesar = {
        "resultado_benchmark_02_Jastrow.log": "02 - Jastrow (Mean Field) + Metropolis",
        "resultado_benchmark_03_RBM.log": "03 - RBM + Metropolis",
        "resultado_benchmark_04_ViT.2.log": "04 - ViT + Metropolis",
        "resultado_benchmark_05_AR.log": "05 - ARNNDense + Metropolis",
        "resultado_benchmark_06_ARNN.log": "06 - ARNNDense + Direct Sampling",
        "resultado_benchmark_06_ARViT.log": "06 - ARViT + Direct Sampling"
    }

    for log_file, title in logs_a_procesar.items():
        ruta_completa = os.path.join(directorio_base, log_file)
        
        if os.path.exists(ruta_completa):
            nombre_base = log_file.replace(".log", "")
            png_name = os.path.join(directorio_base, f"training_{nombre_base}.png")
            plot_benchmark_training(ruta_completa, title, png_name, exact_energy=E_EXACTA)
        else:
            print(f"  [AVISO] Archivo no encontrado: {log_file}. Saltando....")