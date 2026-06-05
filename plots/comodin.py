import os
import sys
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import optax
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Ajuste de rutas para importar tus módulos físicos
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from physics.hamiltonian import get_Hamiltonian
from models.vitB import ARSpinViT_Causal
from physics.utils import BestIterKeeper

# ==============================================================================
# FUNCIONES DE EXTRACCIÓN Y PLOTEO INTEGRADAS (MODO PÓSTER)
# ==============================================================================
def extraer_energias(log_path):
    if not os.path.exists(log_path): return []
    with open(log_path, 'r') as f: text = f.read().strip()
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
        except json.JSONDecodeError: break
    if not data_list: return []
    
    data = data_list[-1]
    energy_dict = data.get("Energy", {})
    e_mean_list = energy_dict.get("Mean", energy_dict.get("mean", energy_dict.get("value", [])))
    return [e.get("real", e.get("Mean", e.get("mean", 0.0))) if isinstance(e, dict) else (e.real if isinstance(e, complex) else float(e)) for e in e_mean_list]

def plot_convergencia_integrada(log_path, output_filename, exact_energy, err_rel, fidelidad):
    energy_mean = extraer_energias(log_path)
    iters = range(len(energy_mean))
    
    label_vmc = (f"Energía VMC\n"
                 f"Error Rel: {err_rel:.2f}" + r" $\times 10^{-4}$" + "\n"
                 f"Fidelidad: {fidelidad:.6f}")

    with plt.rc_context({'font.family': 'serif', 'font.size': 22, 'axes.spines.top': True, 'axes.spines.right': True}):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.plot(iters, energy_mean, color="#5499C7", linewidth=2.5, label=label_vmc)
        ax.axhline(exact_energy, color="#F1948A", linestyle="--", linewidth=2.5, label=f"Energía Exacta ({exact_energy:.4f})")
        
        ax.set_xlabel("Épocas (Iteraciones)", fontsize=24, fontweight='bold')
        ax.set_ylabel(r"Energía $\langle H \rangle$", fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlim(0, len(iters))
        ax.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
        ax.legend(loc="upper right", frameon=True, fontsize=16, edgecolor='#BDC3C7', facecolor='#FDFEFE', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

def plot_autocorrelacion_integrada(E_loc, E_mean, E_var, filename, max_lag=100):
    max_lag = min(max_lag, len(E_loc) - 1)
    lags = np.arange(max_lag)
    autocorr = []
    
    for t in lags:
        if t == 0:
            autocorr.append(1.0)
        else:
            cov_t = np.mean((E_loc[:-t] - E_mean) * (E_loc[t:] - E_mean))
            autocorr.append(cov_t / E_var)
            
    with plt.rc_context({'font.family': 'serif', 'font.size': 22, 'axes.spines.top': True, 'axes.spines.right': True}):
        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(lags, autocorr, marker='o', linestyle='-', color='black', linewidth=2.5, markersize=8, alpha=0.85)
        plt.axhline(0, color='gray', linestyle='--', linewidth=2.0, alpha=0.7)
        
        plt.xlabel("Distancia en la cadena (Lag $t$)", fontsize=24, fontweight='bold')
        plt.ylabel(r"Autocorrelación $C(t)$", fontsize=24, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

# ==============================================================================
# SCRIPT PRINCIPAL DE REPRODUCCIÓN
# ==============================================================================
def reproducir_experimento():
    N = 10
    alpha = 2.5
    J = -4.0
    
    # Directorio de trabajo
    base_dir = "/content/TFG_ARViT/reproduccion"
    os.makedirs(base_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🚀 INICIANDO REPRODUCCIÓN EXACTA: J={J} | alpha={alpha} | Metropolis")
    print(f"{'='*80}\n")
    
    # 1. HAMILTONIANO Y DIAGONALIZACIÓN EXACTA
    print("[+] 1. Diagonalizando el Hamiltoniano...")
    hi = nk.hilbert.Spin(s=1/2, N=N)
    H = get_Hamiltonian(N=N, J=J, alpha=alpha, hilbert=hi)
    H_sparse = H.to_sparse()
    evals, evecs = scipy.sparse.linalg.eigsh(H_sparse, k=1, which="SA")
    E_exact = float(evals[0])
    psi_exact = evecs[:, 0]
    print(f"    -> Energía Exacta calculada: {E_exact:.8f}")

    # 2. CONFIGURACIÓN DEL MODELO Y SAMPLER
    print("\n[+] 2. Configurando ARSpinViT_Causal y MetropolisLocal...")
    model = ARSpinViT_Causal(hilbert=hi, embedding_d=8, n_heads=2, n_blocks=2, n_ffn_layers=1)
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=1, sweep_size=1)
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, n_discard_per_chain=100, seed=42)
    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)
    keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)
    
    log_name = os.path.join(base_dir, f"rep_metropolis_a{alpha}_J{J}")
    log = nk.logging.JsonLog(log_name, save_params=False)

    # 3. ENTRENAMIENTO
    print("\n[+] 3. Iniciando entrenamiento (500 épocas)...")
    start_time = time.time()
    gs.run(n_iter=500, out=log, show_progress=True, callback=keeper.update)
    jax.block_until_ready(vstate.variables)
    tiempo_total = time.time() - start_time
    del log
    
    # 4. EXTRACCIÓN DE MÉTRICAS (Usando el mejor estado)
    print("\n[+] 4. Calculando métricas exactas del mejor estado...")
    vstate.parameters = keeper.best_state.parameters
    
    # 4.1 Energía VQMC y Error Relativo
    E_calc = float(vstate.expect(H).mean.real)
    err_rel = abs((E_calc - E_exact) / E_exact) * 10000
    
    # 4.2 Fidelidad Cuántica
    psi_calc = vstate.to_array()
    psi_calc = psi_calc / jnp.linalg.norm(psi_calc)
    fidelidad = float(jnp.abs(jnp.vdot(psi_calc, psi_exact))**2)
    
    # 4.3 Autocorrelación (Tau al 10%)
    E_loc = np.array(vstate.local_estimators(H).real)[0]
    E_mean = np.mean(E_loc)
    E_var = np.var(E_loc)
    
    tau_10 = "> Max Lag"
    max_lag_search = min(200, len(E_loc) - 1)
    for t in range(1, max_lag_search):
        cov_t = np.mean((E_loc[:-t] - E_mean) * (E_loc[t:] - E_mean))
        c_t = cov_t / E_var
        if c_t <= 0.1:
            tau_10 = t
            break

    # 5. GENERACIÓN DE GRÁFICAS
    print("\n[+] 5. Generando gráficas en formato póster...")
    plot_conv_path = os.path.join(base_dir, "rep_convergencia.png")
    plot_convergencia_integrada(log_name + ".log", plot_conv_path, E_exact, err_rel, fidelidad)
    print(f"    -> Guardada: {os.path.basename(plot_conv_path)}")
    
    plot_auto_path = os.path.join(base_dir, "rep_autocorrelacion.png")
    plot_autocorrelacion_integrada(E_loc, E_mean, E_var, plot_auto_path, max_lag=100)
    print(f"    -> Guardada: {os.path.basename(plot_auto_path)}")

    # 6. RESUMEN FINAL
    print(f"\n{'='*80}")
    print(f"📊 RESUMEN DE LA REPRODUCCIÓN (alpha={alpha}, J={J})")
    print(f"{'='*80}")
    print(f"  • Energía Exacta:       {E_exact:.6f}")
    print(f"  • Energía VQMC (Mín):   {E_calc:.6f}")
    print(f"  • Error Relativo:       {err_rel:.2f} x 10^-4")
    print(f"  • Fidelidad Cuántica F: {fidelidad:.6f}")
    print(f"  • Tau c (10%):          {tau_10}")
    print(f"  • Tiempo de cómputo:    {tiempo_total:.1f} s")
    print(f"{'='*80}\n")
    print(f"📁 Todos los archivos se han guardado en: {base_dir}")

if __name__ == "__main__":
    reproducir_experimento()