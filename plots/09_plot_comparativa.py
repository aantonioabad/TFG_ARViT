import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_comparativa_modelos():
    # Detectar el directorio raíz (asumiendo que el script está en /benchmarks)
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    if os.path.basename(current_dir) == 'benchmarks':
        root_dir = os.path.dirname(current_dir)
    else:
        root_dir = current_dir

    # Diccionario con las etiquetas para la leyenda y las rutas exactas de tus logs
    archivos_logs = {
        "RNN (Metropolis)": os.path.join(root_dir, "resultado_benchmark_03.log"),
        "ARNN (Metropolis)": os.path.join(root_dir, "resultado_benchmark_05.log"),
        "ARNN (Muestreo Directo)": os.path.join(root_dir, "resultado_benchmark_06.log")
    }

    # Colores elegantes para diferenciar bien las curvas
    colores = {
        "RNN (Metropolis)": "#E74C3C",        # Rojo
        "ARNN (Metropolis)": "#F39C12",       # Naranja/Dorado
        "ARNN (Muestreo Directo)": "#2E86C1"  # Azul
    }

    # Valor de energía exacta (Cámbialo si tu baseline de este benchmark es distinto)
    E_exacta = -12.7289  # Valor típico para Ising N=10, h=1 (ajusta según tu caso)

    print("\n" + "="*60)
    print("📊 GENERANDO GRÁFICA COMPARATIVA DE CONVERGENCIA 📊")
    print("="*60 + "\n")

    with plt.rc_context({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': True, 'axes.spines.right': True}):
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        
        # Iterar sobre cada log y añadirlo a la gráfica
        for etiqueta, ruta in archivos_logs.items():
            if not os.path.exists(ruta):
                print(f"  [AVISO] No se encontró el archivo: {os.path.basename(ruta)}")
                continue
                
            with open(ruta, 'r') as f:
                data = json.load(f)
                
            iters = data['Energy']['iters']
            # Extraer la parte real de la energía
            energy_mean = [e.get('real', 0.0) if isinstance(e, dict) else float(np.real(e)) for e in data['Energy']['Mean']]
            
            # Pintar la curva
            ax.plot(iters, energy_mean, label=etiqueta, color=colores[etiqueta], linewidth=1.5, alpha=0.85)
            print(f"  [+] Añadido a la gráfica: {etiqueta}")

        # Añadir la línea de energía exacta como baseline
        ax.axhline(E_exacta, color="black", linestyle="--", linewidth=1.5, label=f"Energía Exacta ({E_exacta})")

        # Formato del gráfico
        ax.set_title("Comparativa de Convergencia: RNN vs ARNN", pad=15, fontweight='bold')
        ax.set_xlabel("Épocas")
        ax.set_ylabel(r"Energía, $\langle H \rangle$")
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
        
        # Leyenda formal
        ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor='#FDFEFE', edgecolor='#BDC3C7')
        
        # Guardar la gráfica en la raíz
        output_path = os.path.join(root_dir, "comparativa_03_05_06.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n  [√] ¡Gráfica guardada con éxito en: {output_path}")

if __name__ == "__main__":
    plot_comparativa_modelos()