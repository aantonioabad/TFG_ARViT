import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def extraer_datos_grafica(log_path):
    """Extrae SOLO las energías del log. Las métricas se fuerzan desde la configuración."""
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
    
    # Buscar energías en el último objeto válido
    data = data_list[-1]
    energy_dict = data.get("Energy", {})
    e_mean_list = energy_dict.get("Mean", energy_dict.get("mean", energy_dict.get("value", [])))
    energies = [e.get("real", e.get("Mean", e.get("mean", 0.0))) if isinstance(e, dict) else (e.real if isinstance(e, complex) else float(e)) for e in e_mean_list]
                
    return energies

def plot_benchmark_training(log_path, benchmark_name, output_filename, exact_energy, err_rel_str, fidelidad_str, max_iters=None):
    print(f"Generando gráfica técnica para: {benchmark_name}...")
    
    energy_mean = extraer_datos_grafica(log_path)
    
    if not energy_mean:
        print(f"  [ERROR] No se pudo leer {log_path} o el archivo está vacío.")
        return

    # --- MODULAR ITERACIONES A PLOTEAR ---
    # Recorta la lista de energías hasta 'max_iters' si se ha especificado un límite
    if max_iters is not None:
        energy_mean = energy_mean[:max_iters]
        
    iters = range(len(energy_mean))
    
    # Construir etiqueta de la leyenda inyectando los datos fijos de la tabla
    label_vmc = (f"Energía VMC\n"
                 f"Error Rel: {err_rel_str}%\n"
                 f"Fidelidad: {fidelidad_str}")

    # ESTÉTICA DE ARTÍCULO CIENTÍFICO (Modo póster gigante)
    with plt.rc_context({
        'font.family': 'serif',
        'font.size': 22,
        'axes.spines.top': True,
        'axes.spines.right': True,
    }):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        color_data = "#5499C7"    # Azul acero pastel
        color_exact = "#F1948A"   # Salmón/Rojo pastel

        # 1. Datos crudos con la nueva leyenda de métricas forzadas
        ax.plot(iters, energy_mean, color=color_data, linewidth=2.5, label=label_vmc)
        
        # 2. Línea exacta gruesa
        ax.axhline(exact_energy, color=color_exact, linestyle="--", linewidth=2.5, 
                    label=f"Energía Exacta ({exact_energy:.4f})")
        
        # ETIQUETAS DE EJE GIGANTES
        ax.set_xlabel("Épocas (Iteraciones)", fontsize=24, fontweight='bold')
        ax.set_ylabel(r"Energía $\langle H \rangle$", fontsize=24, fontweight='bold')
        
        # NÚMEROS DE LOS EJES ENORMES
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        # Ajustar el límite X según si hemos modulado las iteraciones
        if max_iters is not None:
            ax.set_xlim(0, max_iters)
        else:
            ax.set_xlim(0, len(iters))
            
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))

        # Leyenda ajustada para la caja multilínea
        ax.legend(loc="upper right", frameon=True, fontsize=16, edgecolor='#BDC3C7', facecolor='#FDFEFE', framealpha=0.9)
        
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
    
    # ==============================================================================
    # DICCIONARIO DE DATOS FORZADOS (Extraídos directamente de la Tabla 2)
    # Aquí puedes ajustar el "max_iters" para cada modelo como mejor te convenga
    # ==============================================================================
    logs_a_procesar = {
        "resultado_benchmark_02.log": {
            "title": "02 - Jastrow (Mean Field) + Metropolis",
            "err_rel": "0.01", "fidelidad": "0.988098", "max_iters":400 
        },
        "resultado_benchmark_03.log": {
            "title": "03 - RBM + Metropolis",
            "err_rel": "0.09", "fidelidad": "0.996943", "max_iters": 400
        },
        "resultado_benchmark_04.log": {
            "title": "04 - ViT + Metropolis",
            "err_rel": "0.05", "fidelidad": "0.995845", "max_iters": 300
        },
        "resultado_benchmark_05.log": {
            "title": "05 - ARNNDense + Metropolis",
            "err_rel": "0.07", "fidelidad": "0.999247", "max_iters": 80
        },
        "resultado_benchmark_06.log": {
            "title": "06 - ARNNDense + Direct Sampling",
            "err_rel": "0.07", "fidelidad": "0.999465", "max_iters": 80
        },
        "resultado_benchmark_06B.log": {
            "title": "06 - ARViT + Direct Sampling",
            "err_rel": "0.00", "fidelidad": "0.999338", "max_iters": 300
        }
    }

    for log_file, config in logs_a_procesar.items():
        ruta_completa = os.path.join(directorio_base, log_file)
        
        if os.path.exists(ruta_completa):
            nombre_base = log_file.replace(".log", "")
            png_name = os.path.join(directorio_base, f"training_{nombre_base}.png")
            
            plot_benchmark_training(
                log_path=ruta_completa, 
                benchmark_name=config["title"], 
                output_filename=png_name, 
                exact_energy=E_EXACTA,
                err_rel_str=config["err_rel"],
                fidelidad_str=config["fidelidad"],
                max_iters=config["max_iters"] # <-- Parámetro modular aplicado
            )
        else:
            print(f"  [AVISO] Archivo no encontrado: {log_file}. Saltando....")