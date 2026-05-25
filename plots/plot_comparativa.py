import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def generar_todas_las_comparativas_top():
    # 1. Localizar la carpeta general (un nivel por encima de la carpeta 'plots')
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    if os.path.basename(script_dir) == 'plots':
        carpeta_general = os.path.dirname(script_dir)
    else:
        carpeta_general = script_dir

    print(f"[*] Carpeta general detectada: {carpeta_general}")

    # 2. Diccionario apuntando directamente a los nombres de tu captura
    modelos = {
        "Jastrow": {"ruta": os.path.join(carpeta_general, "resultado_benchmark_02.log"), "color": "#34495E"},
        "RBM": {"ruta": os.path.join(carpeta_general, "resultado_benchmark_03.log"), "color": "#E74C3C"},
        "ViT": {"ruta": os.path.join(carpeta_general, "resultado_benchmark_04.log"), "color": "#8E44AD"},
        "ARNN (Metropolis)": {"ruta": os.path.join(carpeta_general, "resultado_benchmark_05.log"), "color": "#F39C12"},
        "ARNN": {"ruta": os.path.join(carpeta_general, "resultado_benchmark_06.log"), "color": "#2E86C1"},
        "ARViT": {"ruta": os.path.join(carpeta_general, "resultado_benchmark_06B.log"), "color": "#27AE60"}
    }

    E_exacta = -12.32525024471575
    E_exacta_label = -12.3253

   
    enfrentamientos = [
        #{"pareja": ["RBM", "ARNN"], "titulo": "Comparativa de Convergencia: RBM vs ARNN", "archivo": "comparativa_03_vs_06.png", "x_max": 900},
        #{"pareja": ["RBM", "ViT"], "titulo": "Comparativa de Convergencia: RBM vs ViT", "archivo": "comparativa_03_vs_04.png", "x_max": 900},
        #{"pareja": ["ViT", "ARNN (Metropolis)"], "titulo": "Comparativa de Convergencia: ViT vs ARNN (Metropolis)", "archivo": "comparativa_04_vs_05.png", "x_max": 900},
        #{"pareja": ["ViT", "ARNN"], "titulo": "Comparativa de Convergencia: ViT vs ARNN", "archivo": "comparativa_04_vs_06.png", "x_max": 350},
        #{"pareja": ["ARNN", "ARViT"], "titulo": "Comparativa de Convergencia: ARNN vs ARViT", "archivo": "comparativa_06_vs_06B.png", "x_max": 350},
        #{"pareja": ["ViT", "ARViT"], "titulo": "Comparativa de Convergencia: ViT vs ARViT", "archivo": "comparativa_04_vs_06B.png", "x_max": 350},
        {"pareja": ["Jastrow", "RBM"], "titulo": "Comparativa de Convergencia: Jastrow vs RBM", "archivo": "comparativa_02_vs_03.png", "x_max": 900},
        {"pareja": ["RBM", "ViT", "ARNN"], "titulo": "Comparativa de Convergencia: RBM vs ViT vs ARNN", "archivo": "comparativa_03_vs_04_vs_05.png", "x_max": 900},
        {"pareja": ["ViT", "ARNN", "ARViT"], "titulo": "Comparativa de Convergencia: ViT vs ARNN vs ARViT", "archivo": "comparativa_05_vs_06_vs_06B.png", "x_max": 900}
    ]

    print("\n" + "="*80)
    print("📊 GENERANDO GRÁFICAS CON MÉTRICAS EN LEYENDA (ERROR Y PEARSON) 📊")
    print("="*80 + "\n")

    for combate in enfrentamientos:
        print(f"[*] Procesando: {combate['archivo']}...")
        
        with plt.rc_context({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': True, 'axes.spines.right': True}):
            fig, ax = plt.subplots(figsize=(9.5, 6), dpi=150) # Un pelín más ancha para que quepa bien la leyenda
            
            for nombre_modelo in combate['pareja']:
                
                id_modelo = "RBM" if nombre_modelo == "RNN" else nombre_modelo
                
                datos_modelo = modelos[id_modelo]
                ruta = datos_modelo['ruta']
                
                if not os.path.exists(ruta):
                    print(f"  [ERROR] No se encontró el log: {os.path.basename(ruta)}")
                    continue
                    
                with open(ruta, 'r') as f:
                    data = json.load(f)
                    
                iters = data['Energy']['iters']
                energy_mean = np.array([e.get('real', 0.0) if isinstance(e, dict) else float(np.real(e)) for e in data['Energy']['Mean']])
                
                # --- CÁLCULO DINÁMICO DE MÉTRICAS PARA LA LEYENDA ---
                best_idx = np.argmin(energy_mean)
                best_energy = energy_mean[best_idx]
                err_relativo = abs((best_energy - E_exacta) / E_exacta)
                
                # Intentar extraer varianza para calcular Pearson dinámicamente
                try:
                    energy_var = np.array([v.get('real', 0.0) if isinstance(v, dict) else float(np.real(v)) for v in data['Energy']['Variance']])
                    best_var = energy_var[best_idx]
                    pearson = np.sqrt(best_var) / abs(best_energy)
                except:
                    pearson = 0.0 # Fallback si el log antiguo no tuviera varianza
                
                # Formateamos la etiqueta de la leyenda de forma compacta y matemática
                label_completo = f"{nombre_modelo} ($\epsilon_r$: {err_relativo:.2%}, $P$: {pearson:.4f})"
                
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
            
            # Ajustamos la leyenda un pelín más pequeña (fontsize=9.5) para que los datos queden perfectamente alineados
            ax.legend(loc="upper right", frameon=True, fontsize=9.5, facecolor='#FDFEFE', edgecolor='#BDC3C7')
            
            output_path = os.path.join(current_dir, combate['archivo'])
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  [√] ¡Guardada con éxito!")

    print("\n[√] ¡Misión cumplida! Gráficas ultraprofesionales listas para compilar.")

if __name__ == "__main__":
    generar_todas_las_comparativas_top()