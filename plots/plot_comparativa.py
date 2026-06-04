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
            "color": "#34495E", "err_str": "0.01%", "pearson_str": "0.0192"
        },
        "RBM": {
            "ruta": os.path.join(carpeta_general, "resultado_benchmark_03.log"), 
            "color": "#E74C3C", "err_str": "0.09%", "pearson_str": "0.0365"
        },
        "ViT": {
            "ruta": os.path.join(carpeta_general, "resultado_benchmark_04.log"), 
            "color": "#8E44AD", "err_str": "0.05%", "pearson_str": "0.0258"
        },
        "ARNN (Metropolis)": {
            "ruta": os.path.join(carpeta_general, "resultado_benchmark_05.log"), 
            "color": "#F39C12", "err_str": "0.07%", "pearson_str": "0.0190"
        },
        "ARNN": {
            "ruta": os.path.join(carpeta_general, "resultado_benchmark_06.log"), 
            "color": "#2E86C1", "err_str": "0.07%", "pearson_str": "0.0166"
        },
        "ARViT": {
            "ruta": os.path.join(carpeta_general, "resultado_benchmark_06B.log"), 
            "color": "#27AE60", "err_str": "0.00%", "pearson_str": "0.01136"
        }
    }

    E_exacta = -12.32525024471575
    E_exacta_label = -12.3253 

    # ==============================================================================
    # AQUÍ VAN TUS ENFRENTAMIENTOS
    
    enfrentamientos = [
        {
            "pareja": ["ARNN (Metropolis)", "ARNN"],
            "archivo": "comparativa_05_vs_06.png",
            "x_max": 80  
        },
        {
            "pareja": ["ViT", "RBM", "ARNN (Metropolis)"],
            "archivo": "comparativa_04_vs_05_vs_03.png",
            "x_max": 80  
        },
        {
            "pareja": ["ViT", "ARNN", "ARViT"],
            "archivo": "comparativa_04_vs_06_vs_06B.png",
            "x_max": 400  #
        },
        {
            "pareja": ["ViT", "ARViT"],
            "archivo": "comparativa_04_vs_06B.png",
            "x_max": 400  # 900 épocas para ver la evolución completa de todos los modelos
        }
        
    ]
    # ==============================================================================

    print("\n" + "="*80)
    print("📊 GENERANDO GRÁFICAS (TAMAÑO DE TEXTO EXTREMO) 📊")
    print("="*80 + "\n")

    for combate in enfrentamientos:
        print(f"[*] Procesando: {combate['archivo']}...")
        
        # Aumentamos el tamaño base de la fuente a 22
        with plt.rc_context({'font.family': 'serif', 'font.size': 22, 'axes.spines.top': True, 'axes.spines.right': True}):
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            
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
                
                err_str = datos_modelo["err_str"]
                pearson_str = datos_modelo["pearson_str"]
                
                label_completo = rf"{nombre_modelo} ($\epsilon_r$: {err_str}, $P$: {pearson_str})"
                
                # Líneas más gruesas (2.5) para que no se pierdan con el texto tan grande
                ax.plot(iters, energy_mean, label=label_completo, color=datos_modelo['color'], linewidth=2.5, alpha=0.85)

            # Línea de energía exacta
            ax.axhline(E_exacta, color="black", linestyle="--", linewidth=2.5, label=f"Energía Exacta ({E_exacta_label:.4f})")

            # ETIQUETAS DE EJE GIGANTES (Tamaño 24)
            ax.set_xlabel("Épocas", fontsize=24, fontweight='bold')
            ax.set_ylabel(r"Energía, $\langle H \rangle$", fontsize=24, fontweight='bold')
            
            # NÚMEROS DE LOS EJES ENORMES (Tamaño 20)
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            ax.set_xlim(0, combate['x_max'])
            ax.set_ylim(E_exacta - 0.05, -10.0) 

            ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
            
            # LEYENDA GIGANTE (Tamaño 18)
            ax.legend(loc="upper right", frameon=True, fontsize=18, facecolor='#FDFEFE', edgecolor='#BDC3C7')
            
            output_path = os.path.join(current_dir, combate['archivo'])
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  [√] ¡Guardada con éxito!")

if __name__ == "__main__":
    generar_todas_las_comparativas_top()