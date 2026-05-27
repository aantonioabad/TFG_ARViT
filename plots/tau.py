import os
import sys
import json
import numpy as np
import jax.numpy as jnp
import netket as nk
import optax
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
base_tfg_dir = "/content/TFG_ARViT"
if base_tfg_dir not in sys.path:
    sys.path.append(base_tfg_dir)

from models.vitB import ARSpinViT_Causal

# ==============================================================================
# HAMILTONIANO J-ALPHA (1D - CADENA)
# ==============================================================================
def get_Hamiltonian_1D_Jalpha(N: int, J: float, alpha: float, h: float = 1.0, hilbert=None):
    if hilbert is None:
        hilbert = nk.hilbert.Spin(s=0.5, N=N)
    
    # [CORREGIDO] Usamos una cadena 1D (Chain) con condiciones periódicas
    graph = nk.graph.Chain(length=N, pbc=True)
    distances = graph.distances()
    H = nk.operator.LocalOperator(hilbert)
    
    sigmax = np.array([[0, 1], [1, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])
    
    # Campo transversal
    for i in range(N):
        H += nk.operator.LocalOperator(hilbert, -h * sigmax, [i])
        
    # Interacción de largo alcance 1D
    for i in range(N):
        for j in range(i + 1, N):
            dist = distances[i][j]
            if dist > 0:
                coupling = J / (dist ** alpha)
                term = coupling * np.kron(sigmaz, sigmaz)
                H += nk.operator.LocalOperator(hilbert, term, [i, j])
                
    return H

# ==============================================================================
# LECTOR ROBUSTO DE LOGS (ENERGÍA Y TAU)
# ==============================================================================
def extraer_datos_log(log_path):
    if not os.path.exists(log_path):
        return [], []
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
            
    if not data_list: return [], []
    data = data_list[-1]
    energy_dict = data.get("Energy", {})
    
    e_mean_list = energy_dict.get("Mean", energy_dict.get("mean", energy_dict.get("value", [])))
    energies = [e.get("real", e.get("Mean", e.get("mean", 0.0))) if isinstance(e, dict) else (e.real if isinstance(e, complex) else float(e)) for e in e_mean_list]
    
    tau_list = energy_dict.get("TauCorr", [])
    taus = [float(t) for t in tau_list]
    
    return energies, taus

# ==============================================================================
# BUCLE PRINCIPAL METROPOLIS 1D
# ==============================================================================
def run_metropolis_jalpha_1d():
    N = 16            # Tamaño de la cadena 1D
    J_val = 1.0       # Interacción
    alpha_val = 2.0   # Parámetro alpha del decaimiento
    
    print(f"\n=========================================================")
    print(f"🎲 METROPOLIS: Modelo 1D J-Alpha (N={N}, α={alpha_val})")
    print(f"=========================================================")
    
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian_1D_Jalpha(N=N, J=J_val, alpha=alpha_val, hilbert=hi)

    model = ARSpinViT_Causal(
        hilbert=hi, embedding_d=8, n_heads=2, n_blocks=2, n_ffn_layers=1
    )

    
    sampler = nk.sampler.MetropolisLocal(
        hi,
        n_chains=1, # Número de exploradores en paralelo
        sweep_size=1   # Muestras que se dejan pasar entre extracciones
    )
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, n_discard_per_chain=100, seed=42)
    
    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

    # Guardar logs directo en la carpeta base
    log_base = os.path.join(base_tfg_dir, f"Metropolis_Jalpha_N{N}_a{alpha_val}")
    logger = nk.logging.JsonLog(log_base, mode="write")

    print("[*] Iniciando entrenamiento VMC con METROPOLIS (500 iteraciones)...")
    gs.run(n_iter=500, out=logger, show_progress=True)
    
    print("\n[+] Entrenamiento terminado. Extrayendo métricas...")
    
    # --- PROCESAMIENTO DE DATOS ---
    log_file_path = log_base + ".log"
    energies, taus = extraer_datos_log(log_file_path)
    
    if taus and energies:
        # Encontrar la mejor iteración
        best_iter = np.argmin(energies)
        best_tau = taus[best_iter]
        best_energy = energies[best_iter]
        
        tau_media = np.mean(taus)
        
        print("\n=========================================================")
        print("📊 ANÁLISIS DE AUTOCORRELACIÓN (METROPOLIS)")
        print("=========================================================")
        print(f"Mejor iteración (Energía mínima)   : Época {best_iter}")
        print(f"Energía en la mejor iteración      : {best_energy:.6f}")
        print(f"--> Tau en la MEJOR iteración      : {best_tau:.4f} <--")
        print(f"Tau promedio durante el proceso    : {tau_media:.4f}")
        print("=========================================================\n")
        
        # Gráfica de la evolución de Tau
        plt.figure(figsize=(10, 6))
        iters = np.arange(len(taus))
        plt.plot(iters, taus, label=rf"Autocorrelación ($\tau$)", color='#d62728', linewidth=2, alpha=0.8)
        plt.axhline(y=tau_media, color='black', linestyle='--', label=f"Media: {tau_media:.2f}")
        
        # Marcar la mejor iteración en la gráfica
        plt.scatter(best_iter, best_tau, color='blue', s=100, zorder=5, 
                    label=f'Mejor Época ({best_iter})\n$\\tau$ = {best_tau:.4f}')
        
        plt.xlabel("Épocas (Iteraciones)", fontsize=13, fontweight='bold')
        plt.ylabel(r"Tiempo de Autocorrelación $\tau$", fontsize=13, fontweight='bold')
        plt.title(f"Evolución Autocorrelación Metropolis - 1D J-Alpha ($\\alpha={alpha_val}$)", fontsize=15, pad=15)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        path_tau = os.path.join(base_tfg_dir, f"autocorr_Metropolis_Jalpha_a{alpha_val}.png")
        plt.savefig(path_tau, dpi=300)
        plt.close()
        
        print(f"[√] Gráfica de autocorrelación guardada en: {path_tau}")
    else:
        print("[X] No se encontraron datos válidos en el log.")

if __name__ == "__main__":
    run_metropolis_jalpha_1d()
                 





