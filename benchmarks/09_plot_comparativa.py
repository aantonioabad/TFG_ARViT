import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_comparativa_limpia(log_files_dict, directorio, output_filename, exact_energy):
    print("Generando comparativa continua con submuestreo...")

    # ESTÉTICA DE ARTÍCULO CIENTÍFICO (LaTeX-like)
    with plt.rc_context({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.spines.top': True,
        'axes.spines.right': True,
    }):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        for filename, info in log_files_dict.items():
            ruta = os.path.join(directorio, filename)
            if not os.path.exists(ruta):
                print(f"  [AVISO] No se encontró {filename}, saltando...")
                continue

            with open(ruta, 'r') as f:
                data = json.load(f)

            iters_full = np.array(data['Energy']['iters'])
            
            energy_full = []
            for e in data['Energy']['Mean']:
                if isinstance(e, dict):
                    energy_full.append(e.get('real', 0.0))
                else:
                    energy_full.append(float(np.real(e)))
            energy_full = np.array(energy_full)

            # --- LA MAGIA DEL SUBMUESTREO ---
            # 1. Cogemos todas las épocas hasta la 400 (pasos de 1 en 1)
            idx_inicio = np.where(iters_full <= 400)[0]
            
            # 2. A partir de la 400, cogemos 1 de cada 25 épocas (pasos grandes)
            idx_final = np.where(iters_full > 400)[0][::25] 
            
            # 3. Unimos los índices
            idx_combinados = np.concatenate((idx_inicio, idx_final))
            
            iters_plot = iters_full[idx_combinados]
            energy_plot = energy_full[idx_combinados]

            # Pintamos la línea con los datos filtrados
            ax.plot(iters_plot, energy_plot, color=info['color'], linewidth=1.5, label=info['label'])

        # LÍNEA EXACTA
        color_exact = "#F1948A" # Salmón pastel
        ax.axhline(exact_energy, color=color_exact, linestyle="--", linewidth=1.8, 
                   label=f"Energía Exacta ({exact_energy:.2f})")

        # ETIQUETAS Y TÍTULOS
        ax.set_title("COMPARATIVA DE ARQUITECTURAS", pad=15)
        ax.set_xlabel("Épocas")
        ax.set_ylabel(r"Parámetro de control, $H$")
        
        # MARCAS DEL EJE X PERSONALIZADAS
        # Alta resolución al principio (0, 100, 200, 300, 400) y baja al final (700, 1000)
        ax.set_xticks([0, 100, 200, 300, 400, 700, 1000])
        ax.set_xlim(0, 1000)

        # CUADRÍCULA
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))

        # LEYENDA
        ax.legend(loc="upper right", frameon=False, fontsize=9)

        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [ÉXITO] Gráfica guardada como '{output_filename}'\n")

if __name__ == "__main__":
    print("\n--- GENERANDO COMPARATIVA DEFINITIVA (EJE ÚNICO) ---\n")
    
    directorio_logs = "/content/drive/MyDrive/TFG_ARViT/graficas y resultados modelos/"
    E_EXACTA = -12.32525024471575
    
    modelos_a_comparar = {
        "resultado_benchmark_02_Jastrow.log": {
            "label": "Jastrow + Metropolis", 
            "color": "#A3E4D7"  # Verde Menta pastel
        },
        "resultado_benchmark_04_ViT.log": {
            "label": "ViT + Metropolis", 
            "color": "#F5CBA7"  # Naranja Melocotón pastel
        },
        "resultado_benchmark_06_ARViT.log": {
            "label": "ARViT + Directo", 
            "color": "#5499C7"  # Azul Acero
        }
    }

    archivo_salida = directorio_logs + "training_Comparativa_Continua.png"
    
    plot_comparativa_limpia(modelos_a_comparar, directorio_logs, archivo_salida, E_EXACTA)