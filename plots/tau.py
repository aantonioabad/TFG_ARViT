import json
import os
import numpy as np

def extraer_autocorrelacion():
    # Ruta donde se guardaron los logs
    drive_dir = "/content/drive/MyDrive/TFG_ARViT/Fase_ J_alpha/"
    
    # Los 6 puntos exactos de tu benchmark
    experimentos = [
        (6.0, 7.0),
        (6.0, -4.0),
        (2.5, -4.0),
        (2.5, 2.75),
        (1.0, -2.0),
        (1.0, 7.0)
    ]

    print("\n" + "="*70)
    print("📊 EXTRACCIÓN DEL TIEMPO DE AUTOCORRELACIÓN (METROPOLIS MCMC) 📊")
    print("="*70)
    print(f"{'J':>6} | {'alpha':>6} | {'τ_c (Última época)':>18} | {'τ_c (Media últimas 50)':>22}")
    print("-" * 70)

    for alpha, J in experimentos:
        log_path = os.path.join(drive_dir, f"resultado_metropolis_alpha{alpha}_J{J}.log")
        
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                data = json.load(f)
            
            try:
                # Extraemos la lista completa de tiempos de autocorrelación
                tau_list = data["Energy"]["TauCorr"]
                
                # Cogemos el valor de la iteración 500
                tau_last = tau_list[-1]
                
                # Calculamos el promedio de la cola (últimas 50 iteraciones)
                tau_avg = np.mean(tau_list[-50:])
                
                print(f"{J:6.2f} | {alpha:6.2f} | {tau_last:18.4f} | {tau_avg:22.4f}")
                
            except KeyError:
                print(f"{J:6.2f} | {alpha:6.2f} | Error: No se encontró 'TauCorr' en el log")
        else:
            print(f"{J:6.2f} | {alpha:6.2f} | Archivo log no encontrado")
            
    print("="*70 + "\n")

if __name__ == "__main__":
    extraer_autocorrelacion()