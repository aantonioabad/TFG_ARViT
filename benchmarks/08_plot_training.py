import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_benchmark_training(log_filename, benchmark_name, output_filename):
    print(f"Generando gráfica de convergencia para: {benchmark_name}...")
    
    # 1. Leer el archivo log (arreglado para evitar el doble .log)
    if log_filename.endswith(".log"):
        log_path = log_filename
    else:
        log_path = f"{log_filename}.log"
        
    if not os.path.exists(log_path):
        print(f"  [ERROR] No se encuentra el archivo {log_path}. Comprueba el nombre exacto.")
        return

    with open(log_path, 'r') as f:
        data = json.load(f)
        
    # 2. Extraer las iteraciones y la energía media
    iters = data['Energy']['iters']
    
    energy_mean = []
    for e in data['Energy']['Mean']:
        if isinstance(e, dict):
            energy_mean.append(e.get('real', 0.0))
        else:
            energy_mean.append(float(np.real(e)))

    # 3. Estilo Profesional para el TFG
    with plt.rc_context({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'grid.color': "#DDDDDD",
        'grid.alpha': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }):
        plt.figure(figsize=(9, 6), dpi=100)

        plt.plot(iters, energy_mean, color="#111111", label="Energía (Loss)", linewidth=1.5)
        
        plt.title(f"Entrenamiento: {benchmark_name}")
        plt.xlabel("Iteración de Entrenamiento (Épocas)")
        
        # ARREGLADO: La 'r' inicial indica a Python que es un texto crudo para LaTeX
        plt.ylabel(r"Energía del Sistema $\langle H \rangle$")
        
        plt.legend(loc="upper right", frameon=True, shadow=True)
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(output_filename, dpi=300)
        plt.close()
        print(f"  [ÉXITO] Gráfica guardada como '{output_filename}'\n")


if __name__ == "__main__":
    print("\n--- GENERANDO GRÁFICAS DE ENTRENAMIENTO DE BENCHMARKS ---\n")
    directorio_logs = "graficas y resultados modelos"
    # Formato: "nombre_del_archivo_sin_extension": "Título para la Gráfica"
    logs_a_procesar = {
        "resultado_benchmark_02_Jastrow.log": "02 - Jastrow (Mean Field) + Metropolis",
        "resultado_benchmark_03_LSTMN.log": "03 - LSTMN + Metropolis",
        "resultado_benchmark_04_ViT.log": "04 - ViT Estándar + Metropolis",
        "resultado_benchmark_05_AR.log": "05 - ARNN + Metropolis",
        "resultado_benchmark_06_ARNN.log": "06 - ARNN Causal + Direct Sampling",
        "resultado_benchmark_06_ARViT.log": "06 - ARViT Causal + Direct Sampling"
    }

    for log_file, title in logs_a_procesar.items():
        png_name = f"training_{log_file}.png"
        plot_benchmark_training(log_file, title, png_name)