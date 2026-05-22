import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_comparativa_modelos():
    # 1. Detección robusta del directorio raíz
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    if os.path.basename(current_dir) in ['benchmarks', 'plots']:
        root_dir = os.path.dirname(current_dir)
    else:
        root_dir = current_dir

    archivos_logs = {
        "RNN (Metropolis)": os.path.join(root_dir, "resultado_benchmark_03.log"),
        "ARNN (Metropolis)": os.path.join(root_dir, "resultado_benchmark_05.log"),
        "ARNN (Muestreo Directo)": os.path.join(root_dir, "resultado_benchmark_06.log")
    }

    colores = {
        "RNN (Metropolis)": "#E74C3C",        # Rojo
        "ARNN (Metropolis)": "#F39C12",       # Naranja/Dorado
        "ARNN (Muestreo Directo)": "#2E86C1"  # Azul
    }

    # NUEVO VALOR DE ENERGÍA EXACTA
    E_exacta = -12.32525024471575
    E_exacta_label = -12.3253  # Redondeado para la leyenda

    print("\n" + "="*60)
    print("📊 GENERANDO GRÁFICA COMPARATIVA DE CONVERGENCIA (ZOOM) 📊")
    print("="*60 + "\n")

    with plt.rc_context({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': True, 'axes.spines.right': True}):
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        
        for etiqueta, ruta in archivos_logs.items():
            if not os.path.exists(ruta):
                print(f"  [AVISO] No se encontró el archivo: {os.path.basename(ruta)}")
                continue
                
            with open(ruta, 'r') as f:
                data = json.load(f)
                
            iters = data['Energy']['iters']
            energy_mean = [e.get('real', 0.0) if isinstance(e, dict) else float(np.real(e)) for e in data['Energy']['Mean']]
            
            ax.plot(iters, energy_mean, label=etiqueta, color=colores[etiqueta], linewidth=1.5, alpha=0.85)
            print(f"  [+] Añadido a la gráfica: {etiqueta}")

        ax.axhline(E_exacta, color="black", linestyle="--", linewidth=1.5, label=f"Energía Exacta ({E_exacta_label:.4f})")

        ax.set_title("Comparativa de Convergencia: RNN vs ARNN", pad=15, fontweight='bold')
        ax.set_xlabel("Épocas")
        ax.set_ylabel(r"Energía, $\langle H \rangle$")
        
        # --- LOS CORTES DE LOS EJES (ZOOM) ESTÁN AQUÍ ---