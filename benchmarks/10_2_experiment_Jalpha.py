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

# Ajuste de rutas para tus módulos
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from physics.hamiltonian import get_Hamiltonian
from models.vitB import ARSpinViT_Causal
from physics.utils import BestIterKeeper

# ==============================================================================
# LECTOR ROBUSTO DE LOGS (ENERGÍA Y TAU) - PROTEGIDO CONTRA NULL
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
    
    # Extraer energía
    e_mean_list = energy_dict.get("Mean", energy_dict.get("mean", energy_dict.get("value", [])))
    energies = [e.get("real", e.get("Mean", e.get("mean", 0.0))) if isinstance(e, dict) else (e.real if isinstance(e, complex) else float(e)) for e in e_mean_list]
    
    # Extraer Autocorrelación (TauCorr) evitando crasheos con None/null
    tau_list = energy_dict.get("TauCorr", [])
    taus = [float(t) if t is not None else 0.0 for t in tau_list]
    
    return energies, taus

# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================
def run_metropolis_sampling_6_points(N=10):
    # Ruta en el repositorio clonado de Colab
    base_dir = "/content/TFG_ARViT"
    os.makedirs(base_dir, exist_ok=True)

    # Solo los 6 puntos exactos para la comparativa (alpha, J)
    experimentos = [
        (6.0, 7.0),
        (6.0, -4.0),
        (2.5, -4.0),
        (2.5, 2.75),
        (1.0, -2.0),
        (1.0, 7.0)
    ]

    summary_table = []

    print(f"\n{'='*95}")
    print(f"🚀 INICIANDO BENCHMARK: METROPOLIS MCMC + FIDELIDAD + TAU (6 PUNTOS)")
    print(f"💾 Destino: {base_dir}")
    print(f"🎲 Sampler: MetropolisLocal")
    print(f"{'='*95}\n")

    for alpha, J in experimentos:
        print(f"\n>>> TRABAJANDO EN: J={J} | alpha={alpha} <<<")
        
        # 1. Hamiltoniano y Energía Exacta (ED)
        hi = nk.hilbert.Spin(s=1/2, N=N)
        H = get_Hamiltonian(N=N, J=J, alpha=alpha, hilbert=hi)
        H_sparse = H.to_sparse()
        evals, evecs = scipy.sparse.linalg.eigsh(H_sparse, k=1, which="SA")
        E_exact = float(evals[0])
        
        print(f"  [+] Energía Exacta: {E_exact:.6f}")

        # 2. Configuración del Modelo
        model = ARSpinViT_Causal(
            hilbert=hi,
            embedding_d=8,
            n_heads=2,
            n_blocks=2,
            n_ffn_layers=1
        )
        
        # Muestreo Metropolis Local
        sampler = nk.sampler.MetropolisLocal(
            hi,
            n_chains=1,    # Número de exploradores en paralelo
            sweep_size=1   # Muestras que se dejan pasar entre extracciones
        )
        
        # Estado variacional (Añadido n_discard_per_chain para el burn-in de Metropolis)
        vstate = nk.vqs.MCState(sampler, model, n_samples=2048, n_discard_per_chain=100, seed=42)
        
        optimizer = optax.adam(learning_rate=0.001)
        gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)
        keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

        # 3. Nombres de archivos
        base_log_name = os.path.join(base_dir, f"resultado_metropolis_alpha{alpha}_J{J}")
        log = nk.logging.JsonLog(base_log_name, save_params=False)

        # 4. Ejecución del Entrenamiento
        print(f"  [+] Entrenando 500 épocas con Metropolis Sampling...")
        start_time = time.time()
        gs.run(n_iter=500, out=log, show_progress=True, callback=keeper.update)
        jax.block_until_ready(vstate.variables)
        
        # Forzamos el cierre del logger para que vuelque todo al disco
        del log 
        
        # 5. EXTRAER TAU DE LA MEJOR ÉPOCA DESDE EL LOG
        log_path = base_log_name + ".log"
        energies_log, taus_log = extraer_datos_log(log_path)
        
        best_tau = 0.0
        best_iter = 0
        if energies_log and taus_log:
            best_iter = int(np.argmin(energies_log))
            best_tau = taus_log[best_iter]
            print(f"  [+] Mejor época en log detectada: Iteración {best_iter} | Tau = {best_tau:.4f}")
            
            # --- GENERAR GRÁFICA INDIVIDUAL DE AUTOCORRELACIÓN ---
            plt.figure(figsize=(10, 6))
            iters = np.arange(len(taus_log))
            plt.plot(iters, taus_log, label=rf"Autocorrelación ($\tau$)", color='#d62728', linewidth=2, alpha=0.8)
            
            tau_media = np.mean(taus_log)
            plt.axhline(y=tau_media, color='black', linestyle='--', label=f"Media: {tau_media:.2f}")
            
            # Marcar la mejor iteración en la gráfica
            plt.scatter(best_iter, best_tau, color='blue', s=100, zorder=5, 
                        label=f'Mejor Época ({best_iter})\n$\\tau$ = {best_tau:.4f}')
            
            plt.xlabel("Épocas (Iteraciones)", fontsize=13, fontweight='bold')
            plt.ylabel(r"Tiempo de Autocorrelación $\tau$", fontsize=13, fontweight='bold')
            plt.title(f"Evolución de Autocorrelación (Metropolis) - J={J}, $\\alpha$={alpha}", fontsize=15, pad=15)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            path_plot_tau = os.path.join(base_dir, f"autocorr_metropolis_alpha{alpha}_J{J}.png")
            plt.savefig(path_plot_tau, dpi=300)
            plt.close()
            print(f"  [√] Gráfica guardada en: {os.path.basename(path_plot_tau)}")

        else:
            print("  [X] Advertencia: No se pudo extraer TauCorr del log.")

        # 6. Cargar mejores parámetros para calcular Energía Final y Fidelidad
        vstate.parameters = keeper.best_state.parameters
        
        E_calc = float(vstate.expect(H).mean.real)
        
        # 7. CALCULAR FIDELIDAD CUÁNTICA EXACTA AL VUELO
        psi_exact = evecs[:, 0]
        psi_arvit = vstate.to_array()
        psi_arvit = psi_arvit / jnp.linalg.norm(psi_arvit)
        
        fidelidad = float(jnp.abs(jnp.vdot(psi_arvit, psi_exact))**2)
        print(f"  [+] Fidelidad Cuántica Calculada: {fidelidad:.6f}")
        
        # Calcular Error Relativo y guardar en el resumen
        err_rel = abs((E_calc - E_exact) / E_exact)
        summary_table.append({
            'J': J,
            'alpha': alpha,
            'E_exact': E_exact,
            'E_calc': E_calc,
            'Fidelidad': fidelidad,
            'Error_Rel': err_rel,
            'Tau': best_tau
        })
        
        # Intentar inyectar en el JSON
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                log_data['Best_Fidelity'] = fidelidad
                log_data['Best_Tau'] = best_tau
                with open(log_path, 'w') as f:
                    json.dump(log_data, f, indent=4)
                print(f"  [💾] Fidelidad y Tau inyectados en el log.")
            except Exception:
                pass 
        
        print(f"  [√] Finalizado en {time.time() - start_time:.1f}s")
        print("-" * 60)

    # =====================================================================
    # TABLA RESUMEN FINAL POR CONSOLA
    # =====================================================================
    print(f"\n{'='*95}")
    print("📊 RESUMEN FINAL DEL BENCHMARK (METROPOLIS MCMC)")
    print(f"{'='*95}")
    print(f"{'J':>6} | {'alpha':>6} | {'E Exacta':>12} | {'E Calculada':>12} | {'Error Rel.':>12} | {'Fidelidad':>10} | {'Tau':>8}")
    print("-" * 95)
    for res in summary_table:
        print(f"{res['J']:6.2f} | {res['alpha']:6.2f} | {res['E_exact']:12.6f} | {res['E_calc']:12.6f} | {res['Error_Rel']:12.2e} | {res['Fidelidad']:10.6f} | {res['Tau']:8.4f}")
    print(f"{'='*95}\n")

if __name__ == "__main__":
    run_metropolis_sampling_6_points(N=10)