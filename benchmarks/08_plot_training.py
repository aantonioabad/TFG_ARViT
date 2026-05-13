import json
import os
import numpy as np
import matplotlib.pyplot as plt

def ema_smooth(scalars, weight=0.85):
    """Aplica un Suavizado Exponencial (estilo TensorBoard) a los datos"""
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_benchmark_training(log_path, benchmark_name, output_filename, exact_energy=None):
    print(f"Generando gráfica HD para: {benchmark_name}...")
    
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

    # Aplicamos el filtro de suavizado
    smoothed_energy = ema_smooth(energy_mean, weight=0.85)

    # Estética Premium
    with plt.rc_context({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'grid.color': "#CCCCCC",
        'grid.alpha': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }):
        # Hacemos la imagen un poco más ancha para formato documento
        plt.figure(figsize=(10, 6), dpi=150)

        # 1. Los datos puros (fondo transparente)
        plt.plot(iters, energy_mean, color="#4A90E2", alpha=0.35, linewidth=1.5, label="Energía VMC (Raw)")
        
        # 2. La tendencia suavizada (línea gruesa principal)
        plt.plot(iters, smoothed_energy, color="#0033A0", linewidth=2.5, label="Tendencia (Suavizada)")
        
        # 3. La línea de Verdad Fundamental (Límite teórico)
        if exact_energy is not None:
            plt.axhline(exact_energy, color="#D0021B", linestyle="--", linewidth=2, 
                        label=f"Energía Exacta ({exact_energy:.4f})")

        plt.title(f"Convergencia de Energía\n{benchmark_name}", pad=15)
        plt.xlabel("Iteración de Entrenamiento (Épocas)")
        plt.ylabel(r"Energía del Sistema $\langle H \rangle$")
        
        # Leyenda estilizada
        plt.legend(loc="upper right", frameon=True, shadow=True, fancybox=True, borderpad=1)
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        
        # Guardamos en alta calidad
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [ÉXITO] Gráfica guardada como '{output_filename}'\n")

if __name__ == "__main__":
    print("\n--- GENERANDO GRÁFICAS DE ENTRENAMIENTO NIVEL DIOS ---\n")
    
    directorio_logs = "/content/drive/MyDrive/TFG_ARViT/graficas y resultados modelos/"
    
    # Hemos añadido la constante mágica para N=10 que calculaste antes
    E_EXACT_10 = -12.784906 
    
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
        # Le añado "_HD" al final para no sobreescribir las anteriores
        png_name = directorio_logs + f"training_{nombre_base}_HD.png" 
        
        # Le pasamos la energía exacta a la función
        plot_benchmark_training(ruta_completa, title, png_name, exact_energy=E_EXACT_10)