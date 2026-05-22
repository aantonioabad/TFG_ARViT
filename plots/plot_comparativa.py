import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def generar_comparativas():
    # 1. Detección robusta del directorio raíz
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    if os.path.basename(current_dir) in ['benchmarks', 'plots']:
        root_dir = os.path.dirname(current_dir)
    else:
        root_dir = current_dir

    # 2. Diccionario maestro con todas las rutas y etiquetas
    modelos = {
        "RNN (Metropolis)": {
            "ruta": os.path.join(root_dir, "resultado_benchmark_03.log"),
            "color": "#E74C3C"  # Rojo
        },
        "ViT (Metropolis)": {
            "ruta": os.path.join(root_dir, "resultado_benchmark_04.log"),
            "color": "#8E44AD"  # Morado
        },
        "ARNN (Muestreo Directo)": {
            "ruta": os.path.join(root_dir, "resultado_benchmark_06.log"),
            "color": "#2E86C1"  # Azul
        }
    }

    # Valor de energía exacta
    E_exacta = -12.32525024471575
    E_exacta_label = -12.3253 

    # 3. Definimos las 3 parejas que queremos enfrentar
    enfrentamientos = [
        {
            "pareja": ["RNN (Metropolis)", "ARNN (Muestreo Directo)"],
            "titulo": "Comparativa de Convergencia: RNN vs ARNN",
            "archivo": "comparativa_03_vs_06.png"
        },
        {
            "pareja": ["RNN (Metropolis)", "ViT (Metropolis)"],
            "titulo": "Comparativa de Convergencia: RNN vs ViT",
            "archivo": "comparativa_03_vs_04.png"
        },
        {
            "pareja": ["ViT (Metropolis)", "ARNN (Muestreo Directo)"],
            "titulo": "Comparativa de Convergencia: ViT vs ARNN",
            "archivo": "comparativa_04_vs_06.png"
        }
    ]

    print("\n" + "="*70)
    print("📊 GENERANDO LAS 3 GRÁFICAS COMPARATIVAS (ZOOM AJUSTADO) 📊")
    print("="*70 + "\n")

    # Bucle principal que genera cada gráfica
    for combate in enfrentamientos:
        print(f"[*] Preparando: {combate['titulo']}...")
        
        with plt.rc_context({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': True, 'axes.spines.right': True}):
            fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
            
            # Pintar las curvas de los modelos en esta pareja
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

            # Línea de energía exacta
            ax.axhline(E_exacta, color="black", linestyle="--", linewidth=1.5, label=f"Energía Exacta ({E_exacta_label:.4f})")

            # Formato y estilo
            ax.set_title(combate['titulo'], pad=15, fontweight='bold')
            ax.set_xlabel("Épocas")
            ax.set_ylabel(r"Energía, $\langle H \rangle$")
            
            # --- LOS CORTES DE LOS EJES (ZOOM) ---
            ax.set_xlim(0, 900)
            ax.set_ylim(E_exacta - 0.05, -10.0) 

            ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
            
            ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor='#FDFEFE', edgecolor='#BDC3C7')
            
            # Guardar gráfica
            output_path = os.path.join(current_dir, combate['archivo'])
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  [√] Guardada en: {output_path}")

    print("\n[√] ¡Misión cumplida! Tienes las 3 gráficas listas.")

if __name__ == "__main__":
    generar_comparativas()