import os
import json
import numpy as np
import matplotlib.pyplot as plt

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
            "titulo": "Evolución del Error Relativo: ViT vs ARViT",
            "archivo": "error_relativo_04_vs_06B.png",
            "x_max": 350
        },
        {
            "pareja": ["ARNN", "ARViT"],
            "titulo": "Evolución del Error Relativo: ARNN vs ARViT",
            "archivo": "error_relativo_06_vs_06B.png",
            "x_max": 350
        }
    ]

    print("\n" + "="*70)
    print("📉 GENERANDO GRÁFICAS DE ERROR RELATIVO LOGARÍTMICO 📉")
    print("="*70 + "\n")

    # Bucle principal para generar cada gráfica
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
                    
                iters = np.array(data['Energy']['iters'])
                # Manejo de números complejos si los hubiera
                energy_mean = np.array([e.get('real', 0.0) if isinstance(e, dict) else float(np.real(e)) for e in data['Energy']['Mean']])
                
                # Fórmula del Error Relativo Absoluto
                error_relativo = np.abs(energy_mean - E_exacta) / np.abs(E_exacta)
                
                # Usamos semilogy para el eje Y logarítmico
                ax.semilogy(iters, error_relativo, label=nombre_modelo, color=datos_modelo['color'], linewidth=1.5, alpha=0.85)

            ax.set_title(combate['titulo'], pad=15, fontweight='bold')
            ax.set_xlabel("Épocas")
            ax.set_ylabel("Error Relativo (Escala Logarítmica)")
            
            # Ajustamos el límite del eje X
            ax.set_xlim(0, combate['x_max'])
            
            # Configuramos la rejilla para que se vea bien en escala logarítmica
            ax.grid(True, which="major", linestyle='-', color='#D5D8DC', linewidth=1.0)
            ax.grid(True, which="minor", linestyle=':', color='#E5E8E8', linewidth=0.8)
            
            ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor='#FDFEFE', edgecolor='#BDC3C7')
            
            output_path = os.path.join(current_dir, combate['archivo'])
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  [√] ¡Guardada en: {output_path}")

    print("\n[√] ¡Misión cumplida! Tienes las gráficas de error relativo listas.")

if __name__ == "__main__":
    generar_errores_relativos()