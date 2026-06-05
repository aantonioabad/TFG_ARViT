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

# ==============================================================================
# GRÁFICA 1: CONVERGENCIA DE ENERGÍA (Azul)
# ==============================================================================
def plot_convergencia_2d(log_path, output_filename, config):
    energy_mean = extraer_energias(log_path)
    if not energy_mean: return

    # --- MODULAR ITERACIONES ---
    max_iters = config.get("max_iters", None)
    if max_iters is not None:
        energy_mean = energy_mean[:max_iters]

    iters = range(len(energy_mean))
    exact_energy = config["E_exact"]
    
    label_vmc = (f"Energía VMC\n"
                 f"Error Rel: {config['err_rel']} %\n"
                 f"Fidelidad: {config['fidelidad']}\n"
                 f"V-score: {config['v_score']}")

    with plt.rc_context({'font.family': 'serif', 'font.size': 22, 'axes.spines.top': True, 'axes.spines.right': True}):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.plot(iters, energy_mean, color="#5499C7", linewidth=2.5, label=label_vmc)
        ax.axhline(exact_energy, color="#F1948A", linestyle="--", linewidth=2.5, label=f"Energía Exacta ({exact_energy:.6f})")
        
        ax.set_xlabel("Épocas (Iteraciones)", fontsize=24, fontweight='bold')
        ax.set_ylabel(r"Energía $\langle H \rangle$", fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        ax.set_xlim(0, len(iters))
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
        ax.legend(loc="upper right", frameon=True, fontsize=22, labelspacing=1.2, edgecolor='#BDC3C7', facecolor='#FDFEFE', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

# ==============================================================================
# GRÁFICA 2: ERROR RELATIVO LOGARÍTMICO (Naranja)
# ==============================================================================
def plot_error_relativo_log(log_path, output_filename, config):
    energy_mean = extraer_energias(log_path)
    if not energy_mean: return

    # --- MODULAR ITERACIONES ---
    max_iters = config.get("max_iters", None)
    if max_iters is not None:
        energy_mean = energy_mean[:max_iters]

    iters = range(len(energy_mean))
    exact_energy = config["E_exact"]
    
    # Cálculo del ERROR RELATIVO en porcentaje (%)
    error_rel = [abs((e - exact_energy) / exact_energy) * 100 for e in energy_mean]

    with plt.rc_context({'font.family': 'serif', 'font.size': 22, 'axes.spines.top': True, 'axes.spines.right': True}):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        color_naranja = "#FF7F0E" 

        ax.plot(iters, error_rel, color=color_naranja, linewidth=2.5, 
                label=r"Error Relativo $\epsilon_{\text{rel}}$ (%)")
        
        ax.set_yscale('log')
        
        ax.set_xlabel("Épocas (Iteraciones)", fontsize=24, fontweight='bold')
        ax.set_ylabel("Error Relativo", fontsize=24, fontweight='bold')
        
        ax.tick_params(axis='both', which='major', labelsize=20)
        
        ax.set_xlim(0, len(iters))
        ax.grid(True, linestyle='--', color='#E5E8E8', linewidth=1.0, alpha=0.7)
        ax.legend(loc="upper right", frameon=True, fontsize=22, labelspacing=1.2, edgecolor='#BDC3C7', facecolor='#FDFEFE', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" GENERANDO GRÁFICAS DE ENERGÍA Y ERROR RELATIVO (RED 2D) ")
    print("="*70)
    
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    if os.path.exists("/content/TFG_ARViT"):
        logs_dir = "/content/TFG_ARViT"
        out_dir = "/content/TFG_ARViT/plots"
    else:
        logs_dir = os.path.abspath(os.path.join(current_dir, "..", "TFG_ARViT"))
        out_dir = current_dir

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n📂 LEYENDO ARCHIVOS LOG DESDE: \n   -> {logs_dir}")
    print(f"💾 GUARDANDO LAS GRÁFICAS (.PNG) EN: \n   -> {out_dir}\n")
    print("-" * 70)

    experimentos_2d = {
        "resultado_benchmark_2D2_ARViT.log": {  
            "base_name": "2d_2x2",
            "E_exact": -4.958881,
            "err_rel": "0.0473",
            "fidelidad": "0.999297",
            "v_score": "0.000719",
            "max_iters": 300  # <--- Límite de iteraciones
        },
        "resultado_benchmark_2D4_ARViT.log": {  
            "base_name": "2d_4x4",
            "E_exact": -25.898045,
            "err_rel": "0.1870",
            "fidelidad": "0.852547",
            "v_score": "0.179719",
            "max_iters": 1000  # <--- Límite de iteraciones
        }
    }

    for log_file, config in experimentos_2d.items():
        ruta_completa = os.path.join(logs_dir, log_file)
        
        if os.path.exists(ruta_completa):
            print(f"[*] Procesando: {log_file} (Límite: {config.get('max_iters', 'Todos los')} pasos)...")
            
            # Gráfica de Energía
            png_energia = os.path.join(out_dir, f"convergencia_energia_{config['base_name']}.png")
            plot_convergencia_2d(ruta_completa, png_energia, config)
            print(f"  [√] Guardada: {png_energia}")
            
            # Gráfica de Error Relativo
            png_error = os.path.join(out_dir, f"error_relativo_{config['base_name']}.png")
            plot_error_relativo_log(ruta_completa, png_error, config)
            print(f"  [√] Guardada: {png_error}\n")
            
        else:
            print(f"  [X] AVISO: Archivo no encontrado -> {log_file}")