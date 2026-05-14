import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def ema_smooth(scalars, weight=0.85):
    """Suavizado para que la curva principal se vea limpia"""
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_benchmark_training(log_path, benchmark_name, output_filename, exact_energy):
    print(f"Generando gráfica estilo Paper para: {benchmark_name}...")
    
    if not os.path.exists(log_path):
        print(f"  [ERROR] No se encuentra el archivo {log_path}.")
        return

    with open(log_path, 'r') as f:
        data = json.load(f)
        
    iters = data['Energy']['iters']
    energy_mean = []
    for e in data['Energy']['Mean']:
        if isinstance(e, dict):
            energy_mean.append(e.get('real', 0.0))
        else:
            energy_mean.append(float(np.real(e)))

    smoothed_energy = ema_smooth(energy_mean, weight=0.85)

    # ESTÉTICA DE ARTÍCULO CIENTÍFICO (LaTeX-like)
    with plt.rc_context({
        'font.family': 'serif', # Fuente tipo académica
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,   # Título más pequeño
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        # Dejamos la caja cerrada (típico de papers de física)
        'axes.spines.top': True,
        'axes.spines.right': True,
    }):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

        # Paleta de colores Pastel
        color_raw = "#A9CCE3"      # Azul pastel muy claro (datos puros)
        color_smooth = "#5499C7"   # Azul pastel medio (tendencia principal)
        color_exact = "#F1948A"    # Salmón/Rojo pastel (límite teórico)

        # 1. Datos crudos (semitransparentes y sin etiqueta para no ensuciar la leyenda)
        ax.plot(iters, energy_mean, color=color_raw, alpha=0.6, linewidth=1.0)
        
        # 2. Tendencia principal (la que sale en la leyenda)
        ax.plot(iters, smoothed_energy, color=color_smooth, linewidth=1.8, label="Energía VMC")
        
        # 3. Línea exacta en color pastel
        ax.axhline(exact_energy, color=color_exact, linestyle="--", linewidth=1.8, 
                    label="Energía Exacta")

        # Título en MAYÚSCULAS y directo al grano
        ax.set_title(benchmark_name.upper(), pad=12)
        
        # Ejes actualizados
        ax.set_xlabel("Épocas")
        ax.set_ylabel(r"Parámetro de control, Valor esperado $\langle H \rangle$")
        
        # Cuadrícula: Líneas continuas y en gris muy suave
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        
        # Aumentar la resolución del eje Y (más marcas numéricas)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))

        # Leyenda: Sin recuadro, letra más pequeña, colocada arriba a la derecha
        ax.legend(loc="upper right", frameon=False, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [ÉXITO] Gráfica guardada como '{output_filename}'\n")

if __name__ == "__main__":
    print("\n--- GENERANDO GRÁFICAS DE ENTRENAMIENTO (ESTILO PAPER) ---\n")
    
    
    directorio_logs = "/content/drive/MyDrive/TFG_ARViT/graficas y resultados modelos/"
    
    
    E_EXACTA = -12.32525024471575
    
    logs_a_procesar = {
        "resultado_benchmark_02_Jastrow.log": "02 - Jastrow (Mean Field) + Metropolis",
        "resultado_benchmark_03_LSTM.log": "03 - LSTM + Metropolis",
        "resultado_benchmark_04_ViT.log": "04 - ViT Estándar + Metropolis",
        "resultado_benchmark_05_AR.log": "05 - ARNNDense + Metropolis",
        "resultado_benchmark_06_ARNN.log": "06 - ARNNDense + Direct Sampling",
        "resultado_benchmark_06_ARViT.log": "06 - ARViT + Direct Sampling",
    }

    for log_file, title in logs_a_procesar.items():
        ruta_completa = directorio_logs + log_file
        nombre_base = log_file.replace(".log", "")
        # Sufijo _Paper para diferenciarlas
        png_name = directorio_logs + f"training_{nombre_base}_Paper.png" 
        
        plot_benchmark_training(ruta_completa, title, png_name, exact_energy=E_EXACTA)