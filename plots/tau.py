import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_energia_convergencia(log_path, exact_energy=None, save_path=None, title="Convergencia de la Energía"):
    if not os.path.exists(log_path):
        print(f"[X] No se encuentra el archivo: {log_path}")
        return

    # 1. Leer el archivo JSON
    with open(log_path, 'r') as f:
        data = json.load(f)

    # 2. Extraer los datos de energía de forma robusta
    energy_dict = data.get("Energy", {})
    if "Mean" in energy_dict:
        e_mean_list = energy_dict["Mean"]
    elif "mean" in energy_dict:
        e_mean_list = energy_dict["mean"]
    elif "value" in energy_dict:
        e_mean_list = energy_dict["value"]
    else:
        print("[X] No se encontraron datos de energía en el log.")
        return

    # Sanear los datos (sacar la parte real)
    energies = []
    for e in e_mean_list:
        if isinstance(e, dict):
            energies.append(e.get("real", e.get("Mean", e.get("mean", 0.0))))
        elif isinstance(e, complex):
            energies.append(e.real)
        else:
            energies.append(float(e))

    iters = np.arange(len(energies))

    # 3. Configuración estética de la gráfica (Estilo TFG)
    plt.figure(figsize=(10, 6))
    
    # Pintar la curva de VMC
    plt.plot(iters, energies, label="Energía Calculada (VMC)", color='#1f77b4', linewidth=2)

    # Pintar la línea de energía exacta (si se proporciona)
    if exact_energy is not None:
        plt.axhline(y=exact_energy, color='#d62728', linestyle='--', linewidth=2, 
                    label=f"Energía Exacta ({exact_energy:.4f})")

    # Detalles de los ejes y leyendas
    plt.xlabel("Épocas (Iteraciones)", fontsize=13, fontweight='bold')
    plt.ylabel("Energía $E$", fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, pad=15)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')
    
    plt.tight_layout()

    # 4. Guardar y mostrar
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[√] Gráfica de alta resolución guardada en: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # --- CONFIGURACIÓN PARA TU ÚLTIMO BENCHMARK 2D ---
    
    # [CORREGIDO] Añadimos TFG_ARViT a la ruta de búsqueda
    archivo_log = "/content/TFG_ARViT/resultado_benchmark_2D_ARViT.log" 
    
    # La energía exacta que te salió en la terminal
    energia_exacta_2D = -29.451812
    
    # Dónde guardar la foto (la guardamos también en la carpeta TFG_ARViT)
    ruta_guardado = "/content/TFG_ARViT/convergencia_energia_2D.png"
    
    plot_energia_convergencia(
        log_path=archivo_log, 
        exact_energy=energia_exacta_2D, 
        save_path=ruta_guardado,
        title="Convergencia de la Energía: ARViT Malla 2D (4x4)"
    )
    
    # Copia de seguridad automática a tu Drive
    os.system(f"cp {ruta_guardado} /content/drive/MyDrive/TFG_ARViT/plots/")