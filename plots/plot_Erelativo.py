import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_error_relativo():
    # 1. Detectar el directorio raíz
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    if os.path.basename(current_dir) == 'benchmarks':
        root_dir = os.path.dirname(current_dir)
    else:
        root_dir = current_dir

    # 2. Archivos a comparar (05 vs 06)
    archivos_logs = {
        "ARNN (Metropolis)": os.path.join(root_dir, "resultado_benchmark_05.log"),
        "ARNN (Muestreo Directo)": os.path.join(root_dir, "resultado_benchmark_06.log")
    }

    colores = {
        "ARNN (Metropolis)": "#F39C12",       # Naranja/Dorado
        "ARNN (Muestreo Directo)": "#2E86C1"  # Azul
    }

    # ¡IMPORTANTE! Pon aquí la energía exacta real de tu sistema para este benchmark
    E_exacta = -12.7289  

    print("\n" + "="*60)
    print("📉 GENERANDO GRÁFICA DE ERROR RELATIVO (05 vs 06) 📉")
    print("="*60 + "\n")

    with plt.rc_context({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': True, 'axes.spines.right': True}):
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        
        for etiqueta, ruta in archivos_logs.items():
            if not os.path.exists(ruta):
                print(f"  [ERROR] No se encontró el log: {os.path.basename(ruta)}")
                continue
                
            with open(ruta, 'r') as f:
                data = json.load(f)
                
            iters = np.array(data['Energy']['iters'])
            # Extraer energía media (parte real)
            energy_mean = np.array([e.get('real', 0.0) if isinstance(e, dict) else float(np.real(e)) for e in data['Energy']['Mean']])
            
            # Calcular el error relativo: |E_medida - E_exacta| / |E_exacta|
            error_relativo = np.abs(energy_mean - E_exacta) / np.abs(E_exacta)
            
            # Ploteamos usando semilogy para que el eje Y sea logarítmico
            ax.semilogy(iters, error_relativo, label=etiqueta, color=colores[etiqueta], linewidth=1.5, alpha=0.85)
            print(f"  [+] Añadida curva: {etiqueta}")

        # Formato de la gráfica
        ax.set_title("Evolución del Error Relativo: Metropolis vs Muestreo Directo", pad=15, fontweight='bold')
        ax.set_xlabel("Épocas")
        ax.set_ylabel("Error Relativo (Escala Logarítmica)")
        
        # Añadimos una cuadrícula adaptada a la escala logarítmica
        ax.grid(True, which="major", linestyle='-', color='#D5D8DC', linewidth=1.0)
        ax.grid(True, which="minor", linestyle=':', color='#E5E8E8', linewidth=0.8)
        
        ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor='#FDFEFE', edgecolor='#BDC3C7')
        
        # Guardar
        output_path = os.path.join(root_dir, "comparativa_error_05_vs_06.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n  [√] ¡Gráfica de error guardada en: {output_path}")

if __name__ == "__main__":
    plot_error_relativo()