import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def extraer_energias(log_path):
    """Extrae la lista de energías medias de un archivo log de NetKet."""
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

def plot_convergencia_2d(log_path, output_filename, config):
    print(f"Procesando: {os.path.basename(log_path)}...")
    
    energy_mean = extraer_energias(log_path)
    
    if not energy_mean:
        print(f"  [ERROR] No se pudo leer {log_path} o el archivo está vacío.")
        return

    iters = range(len(energy_mean))
    exact_energy = config["E_exact"]
    
    # --- AUDITORÍA INTERNA EN CONSOLA ---
    best_energy = min(energy_mean)
    err_rel_calculado = abs((best_energy - exact_energy) / exact_energy) * 100
    
    # --- CONSTRUCCIÓN DE LA LEYENDA (Fuerza los datos de la tabla) ---
    label_vmc = (f"Energía VMC\n"
                 f"Error Rel: {config['err_rel']} %\n"
                 f"Fidelidad: {config['fidelidad']}\n"
                 f"V-score: {config['v_score']}")

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

        # 1. Datos crudos
        ax.plot(iters, energy_mean, color=color_data, linewidth=2.5, label=label_vmc)
        
        # 2. Línea exacta
        ax.axhline(exact_energy, color=color_exact, linestyle="--", linewidth=2.5, 
                    label=f"Energía Exacta ({exact_energy:.6f})")
        
        # ETIQUETAS Y EJES GIGANTES (Sin título)
        ax.set_xlabel("Épocas (Iteraciones)", fontsize=24, fontweight='bold')
        ax.set_ylabel(r"Energía $\langle H \rangle$", fontsize=24, fontweight='bold')
        
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlim(0, len(iters))
            
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))

        # Leyenda ajustada: Se reduce un poco el tamaño a 15 para que quepan las 4 líneas bien
        ax.legend(loc="upper right", frameon=True, fontsize=15, edgecolor='#BDC3C7', facecolor='#FDFEFE', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  [ÉXITO] Guardada como '{os.path.basename(output_filename)}'")
        print(f"          -> Energía mínima detectada en log: {best_energy:.6f}")
        print(f"          -> Error Relativo Calculado internamente: {err_rel_calculado:.4f} %")
        print(f"          -> Error Relativo en Tabla (inyectado): {config['err_rel']} %\n")

if __name__ == "__main__":
    print("\n--- GENERANDO GRÁFICAS 2D (RED 2x2 y 4x4) ---\n")
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    # Ajusta las carpetas según dónde vayas a ejecutar el script
    if os.path.exists("/content/TFG_ARViT"):
        logs_dir = "/content/TFG_ARViT"
        out_dir = "/content/TFG_ARViT/plots"
    else:
        logs_dir = os.path.abspath(os.path.join(current_dir, "..", "TFG_ARViT"))
        out_dir = current_dir

    os.makedirs(out_dir, exist_ok=True)

    # ==============================================================================
    # DICCIONARIO DE EXPERIMENTOS (Datos extraídos de tu tabla)
    # ¡IMPORTANTE! Cambia los nombres de los '.log' por los que tengas en tu carpeta
    # ==============================================================================
    experimentos_2d = {
        "resultado_2d_2x2.log": {  # <--- CAMBIA ESTO por tu nombre real
            "output": "convergencia_2d_2x2.png",
            "E_exact": -4.958881,
            "err_rel": "0.0473",
            "fidelidad": "0.999297",
            "v_score": "0.000719"
        },
        "resultado_2d_4x4.log": {  # <--- CAMBIA ESTO por tu nombre real
            "output": "convergencia_2d_4x4.png",
            "E_exact": -25.898045,
            "err_rel": "0.1870",
            "fidelidad": "0.852547",
            "v_score": "0.179719"
        }
    }

    for log_file, config in experimentos_2d.items():
        ruta_completa = os.path.join(logs_dir, log_file)
        
        if os.path.exists(ruta_completa):
            png_name = os.path.join(out_dir, config["output"])
            plot_convergencia_2d(ruta_completa, png_name, config)
        else:
            print(f"  [AVISO] Archivo no encontrado: {log_file}. ¡Asegúrate de poner el nombre correcto en el diccionario!")