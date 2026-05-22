import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def generar_todas_las_comparativas():
    # 1. Detección robusta del directorio raíz
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    if os.path.basename(current_dir) in ['benchmarks', 'plots']:
        root_dir = os.path.dirname(current_dir)
    else:
        root_dir = current_dir

    # 2. Diccionario con todos los modelos, rutas y colores (Etiquetas limpias)
    modelos = {
        "Jastrow": {
            "ruta": os.path.join(root_dir, "resultado_benchmark_02.log"),
            "color": "#34495E"  # Gris Azulado Oscuro (Baseline)
        },
        "RNN": {
            "ruta": os.path.join(root_dir, "resultado_benchmark_03.log"),
            "color": "#E74C3C"  # Rojo
        },
        "ViT": {
            "ruta": os.path.join(root_dir, "resultado_benchmark_04.log"),
            "color": "#8E44AD"  # Morado
        },
        "ARNN (Metropolis)": {
            "ruta": os.path.join(root_dir, "resultado_benchmark_05.log"),
            "color": "#F39C12"  # Naranja
        },
        "ARNN": {
            "ruta": os.path.join(root_dir, "resultado_benchmark_06.log"),
            "color": "#2E86C1"  # Azul
        },
        "ARViT": {
            "ruta": os.path.join(root_dir, "resultado_benchmark_06B.log"),
            "color": "#27AE60"  # Verde
        }
    }

    # Valor de energía exacta
    E_exacta = -12.32525024471575
    E_exacta_label = -12.3253 

    # 3. Lista con los 7 enfrentamientos y sus límites personalizados en el eje X
    enfrentamientos = [
        {
            "pareja": ["RNN", "ARNN"],
            "titulo": "Comparativa de Convergencia: RNN vs ARNN",
            "archivo": "comparativa_03_vs_06.png",
            "x_max": 950
        },
        {
            "pareja": ["RNN", "ViT"],
            "titulo": "Comparativa de Convergencia: RNN vs ViT",
            "archivo": "comparativa_03_vs_04.png",
            "x_max": 900
        },
        {
            "pareja": ["ViT", "ARNN (Metropolis)"],
            "titulo": "Comparativa de Convergencia: ViT vs ARNN (Metropolis)",
            "archivo": "comparativa_04_vs_05.png",
            "x_max": 350
        },
        {
            "pareja": ["ViT", "ARNN"],
            "titulo": "Comparativa de Convergencia: ViT vs ARNN",
            "archivo": "comparativa_04_vs_06.png",
            "x_max": 350
        },
        {
            "pareja": ["ARNN", "ARViT"],
            "titulo": "Comparativa de Convergencia: ARNN vs ARViT",
            "archivo": "comparativa_06_vs_06B.png",
            "x_max": 350
        },
        {
            "pareja": ["ViT", "ARViT"],
            "titulo": "Comparativa de Convergencia: ViT vs ARViT",
            "archivo": "comparativa_04_vs_06B.png",
            "x_max": 350
        },
        {
            # EL NUEVO ENFRENTAMIENTO A TRES BANDAS
            "pareja": ["Jastrow", "ViT", "ARViT"],
            "titulo": "Comparativa de Convergencia: Jastrow vs ViT vs ARViT",
            "archivo": "comparativa_02_vs_04_vs_06B.png",
            "x_max": 475
        }
    ]

    print("\n" + "="*70)
    print("📊 GENERANDO MASTER COMPLETO DE COMPARATIVAS (7 GRÁFICAS) 📊")
    print("="*70 + "\n")

    # Bucle principal
    for combate in enfrentamientos:
        print(f"[*] Procesando: {combate['archivo']} (Eje X hasta {combate['x_max']})")
        
        with plt.rc_context({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': True, 'axes.spines.right': True}):
            fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
            
            for nombre_modelo in combate['pareja']:
                datos_modelo = modelos[nombre_modelo]
                ruta = datos_modelo['ruta']
                
                if not os.path.exists(ruta):
                    print(f"  [ERROR] No se encontró el log: {os.path.basename(ruta)}")
                    continue
                    
                with open(ruta, 'r') as f:
                    data = json.load(f)
                    
                iters = data['Energy']['iters']
                energy_mean = [e.get('real', 0.0) if isinstance(e, dict) else float(np.real(e)) for e in data['Energy']['Mean']]
                
                ax.plot(iters, energy_mean, label=nombre_modelo, color=datos_modelo['color'], linewidth=1.5, alpha=0.85)

            ax.axhline(E_exacta, color="black", linestyle="--", linewidth=1.5, label=f"Energía Exacta ({E_exacta_label:.4f})")

            ax.set_title(combate['titulo'], pad=15, fontweight='bold')
            ax.set_xlabel("Épocas")
            ax.set_ylabel(r"Energía, $\langle H \rangle$")
            
            # Ajustes de ZOOM dinámicos
            ax.set_xlim(0, combate['x_max'])
            ax.set_ylim(E_exacta - 0.05, -10.0) 

            ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
            
            ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor='#FDFEFE', edgecolor='#BDC3C7')
            
            output_path = os.path.join(current_dir, combate['archivo'])
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  [√] ¡Guardada en: {output_path}")

    print("\n[√] ¡Todo listo! Las 7 gráficas se han guardado con éxito con un diseño impecable.")

if __name__ == "__main__":
    generar_todas_las_comparativas()