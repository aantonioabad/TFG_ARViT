import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def extraer_energias(log_path):
    """Extrae la lista de energías medias de un archivo log de NetKet de forma robusta."""
    if not os.path.exists(log_path):
        return []
    with open(log_path, 'r') as f:
        text = f.read().strip()
        
    decoder = json.JSONDecoder()
    data_list = []
    idx = 0
    while idx < len(text):
        text_substr = text[idx:].lstrip()
        if not text_substr: break
        try:
            obj, next_idx = decoder.raw_decode(text_substr)
            data_list.append(obj)
            idx += next_idx
        except json.JSONDecodeError:
            break
            
    if not data_list: return []
    
    data = data_list[-1]
    energy_dict = data.get("Energy", {})
    e_mean_list = energy_dict.get("Mean", energy_dict.get("mean", energy_dict.get("value", [])))
    energies = [e.get("real", e.get("Mean", e.get("mean", 0.0))) if isinstance(e, dict) else (e.real if isinstance(e, complex) else float(e)) for e in e_mean_list]
                
    return energies

def generar_errores_relativos():
    # 1. Detección robusta del directorio raíz
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    if os.path.basename(current_dir) in ['benchmarks', 'plots']:
        root_dir = os.path.dirname(current_dir)
    else:
        root_dir = current_dir

    # 2. Diccionario con los modelos necesarios y sus colores
    modelos = {
        "ViT": {
            "ruta": os.path.join(root_dir, "resultado_benchmark_04.log"),
            "color": "#8E44AD"  # Morado
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

    # 3. Enfrentamientos solicitados para el error relativo
    enfrentamientos = [
        {
            "pareja": ["ViT", "ARViT"],
            "archivo": "error_relativo_04_vs_06B.png",
            "x_max": 1000
        },
        {
            "pareja": ["ARNN", "ARViT"],
            "archivo": "error_relativo_06_vs_06B.png",
            "x_max": 1000
        }
    ]

    print("\n" + "="*70)
    print("📉 GENERANDO GRÁFICAS DE ERROR RELATIVO LOGARÍTMICO (MODO PÓSTER) 📉")
    print("="*70 + "\n")

    # Bucle principal para generar cada gráfica
    for combate in enfrentamientos:
        print(f"[*] Procesando: {combate['archivo']} (Eje X hasta {combate['x_max']})")
        
        # Estética de Artículo Científico / Póster
        with plt.rc_context({'font.family': 'serif', 'font.size': 22, 'axes.spines.top': True, 'axes.spines.right': True}):
            fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
            
            for nombre_modelo in combate['pareja']:
                datos_modelo = modelos[nombre_modelo]
                ruta = datos_modelo['ruta']
                
                if not os.path.exists(ruta):
                    print(f"  [ERROR] No se encontró el log: {os.path.basename(ruta)}")
                    continue
                    
                energy_mean = extraer_energias(ruta)
                if not energy_mean:
                    continue
                    
                iters = range(len(energy_mean))
                energy_array = np.array(energy_mean)
                
                # Fórmula del Error Relativo Absoluto
                error_relativo = np.abs(energy_array - E_exacta) / np.abs(E_exacta)
                
                # Usamos semilogy para el eje Y logarítmico con líneas gruesas
                ax.semilogy(iters, error_relativo, label=nombre_modelo, color=datos_modelo['color'], linewidth=2.5, alpha=0.85)

            # SIN TÍTULO SUPERIOR (Para cumplir la norma LaTeX)
            
            # Ejes Gigantes
            ax.set_xlabel("Épocas (Iteraciones)", fontsize=24, fontweight='bold')
            ax.set_ylabel("Error Relativo (Escala Log)", fontsize=24, fontweight='bold')
            
            # Números de los ejes grandes
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            # Ajustamos el límite del eje X
            ax.set_xlim(0, combate['x_max'])
            
            # Configuramos la rejilla para que se vea bien en escala logarítmica
            ax.grid(True, which="major", linestyle='-', color='#D5D8DC', linewidth=1.0)
            ax.grid(True, which="minor", linestyle=':', color='#E5E8E8', linewidth=0.8)
            
            # Leyenda grande y espaciada
            ax.legend(loc="upper right", frameon=True, fontsize=20, labelspacing=1.2, facecolor='#FDFEFE', edgecolor='#BDC3C7', framealpha=0.9)
            
            output_path = os.path.join(current_dir, combate['archivo'])
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  [√] ¡Guardada en: {output_path}")

    print("\n[√] ¡Misión cumplida! Tienes las gráficas de error relativo listas y unificadas.")

if __name__ == "__main__":
    generar_errores_relativos()