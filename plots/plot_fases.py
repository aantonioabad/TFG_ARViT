import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def extraer_energias(log_path):
    """Extrae la lista de energías medias de un archivo log de NetKet."""
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

def plot_convergencia(log_path, output_filename, exact_energy, fidelidad_str, err_rel_str):
    print(f"Procesando: {os.path.basename(log_path)}...")
    
    energy_mean = extraer_energias(log_path)
    
    if not energy_mean:
        print(f"  [ERROR] No se pudo leer {log_path} o el archivo está vacío.")
        return

    iters = range(len(energy_mean))
    
    best_energy = min(energy_mean)
    
    # Mantenemos el cálculo interno solo para imprimirlo por consola y auditar
    err_rel_calculado = abs((best_energy - exact_energy) / exact_energy) * 10000
    
    # Modificamos la leyenda para forzar el string exacto de la tabla en formato %
    label_vmc = (f"Energía VMC\n"
                 f"Error Rel: {err_rel_str}%\n"
                 f"Fidelidad: {fidelidad_str}")

    with plt.rc_context({
        'font.family': 'serif',
        'font.size': 22,
        'axes.spines.top': True,
        'axes.spines.right': True,
    }):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        color_data = "#5499C7"    
        color_exact = "#F1948A"   

        ax.plot(iters, energy_mean, color=color_data, linewidth=2.5, label=label_vmc)

        ax.axhline(exact_energy, color=color_exact, linestyle="--", linewidth=2.5, 
                    label=f"Energía Exacta ({exact_energy:.4f})")
        
        ax.set_xlabel("Épocas (Iteraciones)", fontsize=24, fontweight='bold')
        ax.set_ylabel(r"Energía $\langle H \rangle$", fontsize=24, fontweight='bold')
        
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlim(0, len(iters))
            
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))

        ax.legend(loc="upper right", frameon=True, fontsize=20, labelspacing=1.2, edgecolor='#BDC3C7', facecolor='#FDFEFE', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  [ÉXITO] Guardada como '{os.path.basename(output_filename)}'")
        print(f"          -> Energía mínima: {best_energy:.8f}")
        print(f"          -> Error Relativo Calculado (auditoría): {err_rel_calculado:.2f} x 10^-4\n")

if __name__ == "__main__":
    print("\n--- GENERANDO GRÁFICAS Y AUDITANDO ERROR RELATIVO ---\n")
    
    # Forzamos a que lea la carpeta actual (donde están el script y los logs)
    logs_dir = "."
    out_dir = "."
    
    print(f"[*] Buscando logs en: {os.path.abspath(logs_dir)}")
    print(f"[*] Guardando gráficas en: {os.path.abspath(out_dir)}\n")

    # (Aquí va la lista de la variable "experimentos" exactamente igual que la tenías)
    experimentos = [
        {"log": "resultado_direct_alpha1.0_J-2.0.log", "E_exact": -44.239541, "fidelidad": "0.993972", "err_rel": "0.04"},
        {"log": "resultado_metropolis_alpha1.0_J-2.0.log", "E_exact": -44.239541, "fidelidad": "0.998112", "err_rel": "0.12"},
        
        {"log": "resultado_direct_alpha2.5_J-4.0.log", "E_exact": -51.733292, "fidelidad": "0.993284", "err_rel": "0.05"},
        {"log": "resultado_metropolis_alpha2.5_J-4.0.log", "E_exact": -51.733292, "fidelidad": "0.569480", "err_rel": "0.10"},
        
        {"log": "resultado_direct_alpha6.0_J-4.0.log", "E_exact": -41.307541, "fidelidad": "0.999374", "err_rel": "0.00"},
        {"log": "resultado_metropolis_alpha6.0_J-4.0.log", "E_exact": -41.307541, "fidelidad": "0.576599", "err_rel": "0.36"},
        
        {"log": "resultado_direct_alpha1.0_J7.0.log", "E_exact": -48.362404, "fidelidad": "0.963167", "err_rel": "0.07"},
        {"log": "resultado_metropolis_alpha1.0_J7.0.log", "E_exact": -48.362404, "fidelidad": "0.945039", "err_rel": "0.58"},
        
        {"log": "resultado_direct_alpha2.5_J2.75.log", "E_exact": -24.859957, "fidelidad": "0.962598", "err_rel": "0.05"},
        {"log": "resultado_metropolis_alpha2.5_J2.75.log", "E_exact": -24.859957, "fidelidad": "0.831200", "err_rel": "1.22"},
        
        {"log": "resultado_direct_alpha6.0_J7.0.log", "E_exact": -69.350307, "fidelidad": "0.997883", "err_rel": "0.03"},
        {"log": "resultado_metropolis_alpha6.0_J7.0.log", "E_exact": -69.350307, "fidelidad": "0.706565", "err_rel": "0.13"}
    ]

    for exp in experimentos:
        ruta_completa = os.path.join(logs_dir, exp["log"])
        
        if os.path.exists(ruta_completa):
            nombre_base = exp["log"].replace(".log", "")
            png_name = os.path.join(out_dir, f"training_{nombre_base}.png")
            
            plot_convergencia(
                log_path=ruta_completa, 
                output_filename=png_name, 
                exact_energy=exp["E_exact"],
                fidelidad_str=exp["fidelidad"],
                err_rel_str=exp["err_rel"]
            )
        else:
            print(f"  [AVISO] Archivo no encontrado: {exp['log']}. Saltando....")