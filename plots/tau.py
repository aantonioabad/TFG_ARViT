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
os.chdir(base_tfg_dir)

from models.vitB import ARSpinViT_Causal

# ==============================================================================
# HAMILTONIANO J-ALPHA (1D)
# ==============================================================================
def get_Hamiltonian_1D_Jalpha(N: int, J: float, alpha: float, h: float = 1.0, hilbert=None):
    if hilbert is None:
        hilbert = nk.hilbert.Spin(s=0.5, N=N)
    
    graph = nk.graph.Chain(length=N, pbc=True)
    distances = graph.distances()
    H = nk.operator.LocalOperator(hilbert)
    
    sigmax = np.array([[0, 1], [1, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])
    
    for i in range(N):
        H += nk.operator.LocalOperator(hilbert, -h * sigmax, [i])
        
    for i in range(N):
        for j in range(i + 1, N):
            dist = distances[i][j]
            if dist > 0:
                coupling = J / (dist ** alpha)
                term = coupling * np.kron(sigmaz, sigmaz)
                H += nk.operator.LocalOperator(hilbert, term, [i, j])
                
    return H

# ==============================================================================
# LECTOR ROBUSTO DE LOGS
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
# BUCLE PRINCIPAL (BARRIDO DE EXPERIMENTOS)
# ==============================================================================
def run_metropolis_sweep():
    N = 10
    N_ITER = 500  # Iteraciones por punto
    
    # Tus 6 puntos exactos para la comparativa
    experimentos = [
        (6.0, 7.0),
        (6.0, -4.0),
        (2.5, -4.0),
        (2.5, 2.75),
        (1.0, -2.0),
        (1.0, 7.0)
    ]
    
    resultados_tau = []
    etiquetas = []

    print(f"\n=========================================================")
    print(f"🚀 INICIANDO BARRIDO J-ALPHA (METROPOLIS) - {len(experimentos)} PUNTOS")
    print(f"=========================================================")

    for i, (J_val, alpha_val) in enumerate(experimentos, 1):
        print(f"\n---> [Punto {i}/{len(experimentos)}] Entrenando J = {J_val}, α = {alpha_val} ...")
        
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

        # Nombre de archivo único para cada punto
        log_name = f"Metropolis_J{J_val}_a{alpha_val}"
        log_base = os.path.join(base_tfg_dir, log_name)
        logger = nk.logging.JsonLog(log_base, mode="write")

        # Ejecutamos el entrenamiento silenciado (show_progress=False para no ensuciar la terminal)
        gs.run(n_iter=N_ITER, out=logger, show_progress=False)
        
        # Extraemos la mejor iteración
        energies, taus = extraer_datos_log(log_base + ".log")
        if energies and taus:
            best_iter = np.argmin(energies)
            best_tau = taus[best_iter]
            
            print(f"     [√] Completado. Mejor época: {best_iter} | Tau = {best_tau:.4f}")
            resultados_tau.append(best_tau)
            etiquetas.append(f"J={J_val}\n$\\alpha$={alpha_val}")
        else:
            print(f"     [X] Error leyendo log de J={J_val}, α={alpha_val}")
            resultados_tau.append(0.0)
            etiquetas.append(f"J={J_val}\n$\\alpha$={alpha_val}\n(Error)")

    # ==============================================================================
    # RESUMEN FINAL Y GRÁFICA COMPARATIVA
    # ==============================================================================
    print("\n=========================================================")
    print("📊 RESUMEN DE AUTOCORRELACIÓN (MEJOR ÉPOCA) - METROPOLIS")
    print("=========================================================")
    for (J_val, alpha_val), tau in zip(experimentos, resultados_tau):
        print(f"Punto J={J_val:^5}, α={alpha_val:^5}  -->  Tau = {tau:.4f}")
    print("=========================================================\n")

    # Gráfica de Barras Comparativa
    plt.figure(figsize=(10, 6))
    bars = plt.bar(etiquetas, resultados_tau, color='#d62728', alpha=0.8, edgecolor='black')
    
    # Añadir el número exacto encima de cada barra
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(resultados_tau)*0.01), 
                 f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.ylabel(r"Tiempo de Autocorrelación $\tau$ (Mejor Época)", fontsize=13, fontweight='bold')
    plt.title("Impacto del Diagrama de Fases (J, $\\alpha$) en Metropolis", fontsize=15, pad=15)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    path_plot = os.path.join(base_tfg_dir, "comparativa_tau_Metropolis_Jalpha.png")
    plt.savefig(path_plot, dpi=300)
    plt.close()
    
    print(f"[√] Gráfica comparativa guardada en: {path_plot}")

if __name__ == "__main__":
    run_metropolis_sweep()





