import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

def plot_comparativa_broken_axis(log_files_dict, directorio, output_filename, exact_energy):
    print("Generando comparativa avanzada (Eje Roto) para la memoria...")

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
        # Creamos la figura y partimos el espacio (Ratio 3 a 1)
        # El 75% de la imagen será para las iteraciones 0-300, el 25% para 300-1000
        fig = plt.figure(figsize=(9, 5), dpi=150)
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)

        # Iteramos y dibujamos cada modelo en ambos paneles
        for filename, info in log_files_dict.items():
            ruta = os.path.join(directorio, filename)
            if not os.path.exists(ruta):
                print(f"  [AVISO] No se encontró {filename}, saltando...")
                continue

            with open(ruta, 'r') as f:
                data = json.load(f)

            iters = data['Energy']['iters']
            energy_mean = []
            for e in data['Energy']['Mean']:
                if isinstance(e, dict):
                    energy_mean.append(e.get('real', 0.0))
                else:
                    energy_mean.append(float(np.real(e)))

            # ax1 tiene el label para la leyenda, ax2 no lo necesita
            ax1.plot(iters, energy_mean, color=info['color'], linewidth=1.3, label=info['label'])
            ax2.plot(iters, energy_mean, color=info['color'], linewidth=1.3)

        # LÍNEA EXACTA
        color_exact = "#F1948A" # Salmón pastel
        ax1.axhline(exact_energy, color=color_exact, linestyle="--", linewidth=1.8, label=f"Energía Exacta ({exact_energy:.2f})")
        ax2.axhline(exact_energy, color=color_exact, linestyle="--", linewidth=1.8)

        # APLICAMOS EL ZOOM Y LA COMPRESIÓN
        ax1.set_xlim(0, 300)
        ax2.set_xlim(300, 1000)

        # EFECTO DE EJE ROTO (Ocultar bordes interiores)
        ax1.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax1.yaxis.tick_left()
        ax2.tick_params(labelleft=False, left=False) # Quita los números del eje Y de la derecha

        # MARCAS DIAGONALES (//)
        d = .015 # Tamaño de las marcas de corte
        kwargs = dict(transform=ax1.transAxes, color='#999999', clip_on=False, linewidth=1.2)
        ax1.plot((1-d, 1+d), (-d, +d), **kwargs)     # Abajo derecha
        ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)   # Arriba derecha

        kwargs.update(transform=ax2.transAxes)
        d_x = d * 3 # Ajuste geométrico por el ancho del panel derecho
        ax2.plot((-d_x, +d_x), (-d, +d), **kwargs)   # Abajo izquierda
        ax2.plot((-d_x, +d_x), (1-d, 1+d), **kwargs) # Arriba izquierda

        # ETIQUETAS Y TÍTULOS
        ax1.set_ylabel(r"Parámetro de control, $H$")
        
        # Centramos el texto "Épocas" para que abarque los dos gráficos
        ax1.set_xlabel("Épocas")
        ax1.xaxis.set_label_coords(0.66, -0.1) 

        # Título principal
        fig.suptitle("COMPARATIVA DE ARQUITECTURAS", fontsize=11, fontweight='bold', y=0.96)

        # CUADRÍCULAS (Gris pastel continuas)
        ax1.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        ax2.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        
        # Resolución máxima en el eje Y
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=12))

        # LEYENDA (Sin caja, tamaño pequeño)
        ax1.legend(loc="upper right", frameon=False, fontsize=9)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15) # Espacio extra abajo
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [ÉXITO] Gráfica guardada como '{output_filename}'\n")

if __name__ == "__main__":
    print("\n--- GENERANDO COMPARATIVA DEFINITIVA ---\n")
    
    directorio_logs = "/content/drive/MyDrive/TFG_ARViT/graficas y resultados modelos/"
    E_EXACTA = -12.32525024471575
    
    # Diccionario con los 3 modelos clave y sus colores pastel asignados
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
            "color": "#5499C7"  # Azul Acero (Más nítido para resaltar)
        }
    }

    archivo_salida = directorio_logs + "training_Comparativa_BrokenAxis.png"
    
    plot_comparativa_broken_axis(modelos_a_comparar, directorio_logs, archivo_salida, E_EXACTA)