import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def generar_todas_las_comparativas_top():
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    if os.path.basename(current_dir) in ['plots', 'benchmarks']:
        carpeta_general = os.path.dirname(current_dir)
    else:
        carpeta_general = current_dir

    print(f"[*] Carpeta raíz detectada para buscar logs: {carpeta_general}")

    # 2. Diccionario con RUTAS y VALORES EXACTOS DE TU TABLA
    modelos = {
        "Jastrow": {
            "ruta": os.path.join(carpeta_general, "resultado_benchmark_02.log"), 
            "color": "#34495E", "err_str": "0.01%", "pearson_str": "0.0105"
        },
        "RBM": {
            "ruta": os.path.join(carpeta_general, "resultado_benchmark_03.log"), 
            "color": "#E74C3C", "err_str": "0.09%", "pearson_str": "0.0365"
        },
        "ViT": {
            "ruta": os.path.join(carpeta_general, "resultado_benchmark_04.log"), 
            "color": "#8E44AD", "err_str": "0.00%", "pearson_str": "0.0090"
        },
        "ARNN (Metropolis)": {
            "ruta": os.path.join(carpeta_general, "resultado_benchmark_05.log"), 
            "color": "#F39C12", "err_str": "0.02%", "pearson_str": "0.0116"
        },
        "ARNN": {
            "ruta": os.path.join(carpeta_general, "resultado_benchmark_06.log"), 
            "color": "#2E86C1", "err_str": "0.07%", "pearson_str": "0.0166"
        },
        "ARViT": {
            "ruta": os.path.join(carpeta_general, "resultado_benchmark_06B.log"), 
            "color": "#27AE60", "err_str": "0.00%", "pearson_str": "0.0114"
        }
    }

    E_exacta = -12.32525024471575
    E_exacta_label = -12.3253 

    # ==============================================================================
    # AQUÍ VAN TUS ENFRENTAMIENTOS (PON LOS QUE NECESITES)
    enfrentamientos = [
        {"pareja": ["Jastrow", "RBM"], "titulo": "Comparativa: Jastrow vs RBM", "archivo": "comparativa_02_vs_03.png", "x_max": 450},
        {"pareja": ["RBM", "ViT", "ARNN (Metropolis)"], "titulo": "Comparativa Muestreo Metropolis: RBM vs ViT vs ARNN", "archivo": "comparativa_03_vs_04_vs_05.png", "x_max": 450},
        {"pareja": ["ARNN (Metropolis)", "ARNN", "ARViT"], "titulo": "Comparativa de Convergencia: Modelos AR", "archivo": "comparativa_05_vs_06_vs_06B.png", "x_max": 300}
    ]
    # ==============================================================================

    print("\n" + "="*80)
    print("📊 GENERANDO GRÁFICAS (VALORES FORZADOS DESDE TABLA) 📊")
    print("="*80 + "\n")

    for combate in enfrentamientos:
        print(f"[*] Procesando: {combate['archivo']}...")
        
        with plt.rc_context({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': True, 'axes.spines.right': True}):
            fig, ax = plt.subplots(figsize=(9.5, 6), dpi=150)
            
            for nombre_modelo in combate['pareja']:
                datos_modelo = modelos[nombre_modelo]
                ruta = datos_modelo['ruta']
                
                if not os.path.exists(ruta):
                    print(f"  [ERROR] No se encontró el log en: {ruta}")
                    continue
                    
                with open(ruta, 'r') as f:
                    data = json.load(f)
                    
                iters = data['Energy']['iters']
                energy_mean = np.array([e.get('real', 0.0) if isinstance(e, dict) else float(np.real(e)) for e in data['Energy']['Mean']])
                
                # RECOGER VALORES EXACTOS FIJADOS EN EL DICCIONARIO
                err_str = datos_modelo["err_str"]
                pearson_str = datos_modelo["pearson_str"]
                
                # Generar la etiqueta con formato LaTeX
                label_completo = rf"{nombre_modelo} ($\epsilon_r$: {err_str}, $P$: {pearson_str})"
                
                ax.plot(iters, energy_mean, label=label_completo, color=datos_modelo['color'], linewidth=1.5, alpha=0.85)

            # Línea de energía exacta
            ax.axhline(E_exacta, color="black", linestyle="--", linewidth=1.5, label=f"Energía Exacta ({E_exacta_label:.4f})")

            ax.set_title(combate['titulo'], pad=15, fontweight='bold')
            ax.set_xlabel("Épocas")
            ax.set_ylabel(r"Energía, $\langle H \rangle$")
            
            ax.set_xlim(0, combate['x_max'])
            ax.set_ylim(E_exacta - 0.05, -10.0) 

            ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
            
            ax.legend(loc="upper right", frameon=True, fontsize=9.5, facecolor='#FDFEFE', edgecolor='#BDC3C7')
            
            output_path = os.path.join(current_dir, combate['archivo'])
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  [√] ¡Guardada con éxito!")

if __name__ == "__main__":
    generar_todas_las_comparativas_top()