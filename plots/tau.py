import json
import os
import numpy as np

def extraer_autocorrelacion_mejor_epoca():
    # Ruta de los logs en tu Drive
    drive_dir = "/content/drive/MyDrive/TFG_ARViT/Fase_ J_alpha/"
    
    # Los 6 puntos de tu benchmark
    experimentos = [
        (6.0, 7.0),
        (6.0, -4.0),
        (2.5, -4.0),
        (2.5, 2.75),
        (1.0, -2.0),
        (1.0, 7.0)
    ]

    print("\n" + "="*85)
    print("📊 TIEMPO DE AUTOCORRELACIÓN EN LA MEJOR ÉPOCA (BEST ITER KEEPER) 📊")
    print("="*85)
    print(f"{'J':>6} | {'alpha':>6} | {'Mejor Época':>12} | {'E Mínima':>14} | {'τ_c (Mejor Época)':>20}")
    print("-" * 85)

    for alpha, J in experimentos:
        log_path = os.path.join(drive_dir, f"resultado_metropolis_alpha{alpha}_J{J}.log")
        
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                data = json.load(f)
            
            try:
                # 1. Extraemos el histórico de energías
                e_mean_list = data["Energy"]["mean"]
                
                # Saneamos el formato (por si JAX/NetKet lo guardó como dict con parte real)
                energies = []
                for e in e_mean_list:
                    if isinstance(e, dict):
                        energies.append(e.get("real", e.get("mean", 0.0)))
                    elif isinstance(e, complex):
                        energies.append(e.real)
                    else:
                        energies.append(float(e))
                
                # 2. Encontramos el índice de la mejor época (mínima energía)
                best_idx = int(np.argmin(energies))
                best_energy = energies[best_idx]
                
                # 3. Extraemos el TauCorr de esa misma época exacta
                tau_list = data["Energy"]["TauCorr"]
                best_tau = tau_list[best_idx]
                
                print(f"{J:6.2f} | {alpha:6.2f} | {best_idx:12d} | {best_energy:14.6f} | {best_tau:20.4f}")
                
            except KeyError as e:
                print(f"{J:6.2f} | {alpha:6.2f} | Error: No se encontró la clave {e} en el log")
        else:
            print(f"{J:6.2f} | {alpha:6.2f} | Archivo log no encontrado")
            
    print("="*85 + "\n")

if __name__ == "__main__":
    extraer_autocorrelacion_mejor_epoca()