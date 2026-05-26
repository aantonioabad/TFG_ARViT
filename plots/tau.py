import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_energia_convergencia(log_path, exact_energy=None, save_path=None, title="Convergencia de la Energía"):
    if not os.path.exists(log_path):
        print(f"[X] No se encuentra el archivo: {log_path}")
        return

    # Extracción robusta de JSONs concatenados
    with open(log_path, 'r') as f:
        text = f.read().strip()
        
    decoder = json.JSONDecoder()
    data_list = []
    idx = 0
    
    while idx < len(text):
        text_substr = text[idx:].lstrip()
        if not text_substr:
            break
        try:
            obj, next_idx = decoder.raw_decode(text_substr)
            data_list.append(obj)
            idx += next_idx
        except json.JSONDecodeError:
            break
            
    if not data_list:
        print("[X] Imposible extraer ningún dato válido del log.")
        return
        
    data = data_list[-1]

    # Procesamiento de la energía
    energy_dict = data.get("Energy", {})
    e_mean_list = energy_dict.get("Mean", energy_dict.get("mean", energy_dict.get("value", [])))

    if not e_mean_list:
        print("[X] No se encontraron datos de energía en la última ejecución.")
        return

    energies = [e.get("real", e.get("Mean", e.get("mean", 0.0))) if isinstance(e, dict) else (e.real if isinstance(e, complex) else float(e)) for e in e_mean_list]
    iters = np.arange(len(energies))

    # Generación de la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(iters, energies, label="Energía Calculada (VMC)", color='#1f77b4', linewidth=2)

    if exact_energy is not None:
        plt.axhline(y=exact_energy, color='#d62728', linestyle='--', linewidth=2, 
                    label=f"Energía Exacta ({exact_energy:.4f})")

    plt.xlabel("Épocas (Iteraciones)", fontsize=13, fontweight='bold')
    plt.ylabel("Energía $E$", fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, pad=15)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[√] Gráfica de alta resolución guardada en: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    archivo_log = "/content/TFG_ARViT/resultado_benchmark_2D_ARViT.log" 
    energia_exacta_2D = -29.451812
    ruta_guardado = "/content/TFG_ARViT/convergencia_energia_2D.png"
    
    plot_energia_convergencia(
        log_path=archivo_log, 
        exact_energy=energia_exacta_2D, 
        save_path=ruta_guardado,
        title="Convergencia de la Energía: ARViT Malla 2D (4x4)"
    )
    
    os.system(f"cp {ruta_guardado} /content/drive/MyDrive/TFG_ARViT/plots/")