import os
import sys
import time
import json
import jax
import jax.numpy as jnp
import numpy as np  # <-- Añadido para la autocorrelación
import netket as nk
import optax
import scipy.sparse.linalg

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from physics.hamiltonian import get_Hamiltonian
from models.ARViT import ARSpinViT_Causal
# Asumo que plot_markov_autocorrelation está en utils, si no, ajusta este import:
from physics.utils import BestIterKeeper, plot_markov_autocorrelation 

def run_direct_sampling_6_points(N=10):
    # Ruta principal en tu Drive (sin subcarpetas)
    drive_dir = "/content/drive/MyDrive/TFG_ARViT/Fase_ J_alpha/"
    os.makedirs(drive_dir, exist_ok=True)

    experimentos = [
        (6.0, 7.0),
        (6.0, -4.0),
        (2.5, -4.0),
        (2.5, 2.75),
        (1.0, -2.0),
        (1.0, 7.0)
    ]

    print(f"\n{'='*80}")
    print(f"🚀 INICIANDO BENCHMARK: DIRECT SAMPLING + MÉTRICAS + AUTOCORRELACIÓN")
    print(f"💾 Destino: {drive_dir}")
    print(f"🎲 Sampler: ARDirectSampler (Muestreo Directo/Autorregresivo)")
    print(f"{'='*80}\n")

    for alpha, J in experimentos:
        print(f"\n>>> TRABAJANDO EN: J={J} | alpha={alpha} <<<")
        
        # 1. Hamiltoniano y Energía Exacta (ED)
        hi = nk.hilbert.Spin(s=1/2, N=N)
        H = get_Hamiltonian(N=N, J=J, alpha=alpha, hilbert=hi)
        H_sparse = H.to_sparse()
        evals, evecs = scipy.sparse.linalg.eigsh(H_sparse, k=1, which="SA")
        E_exact = float(evals[0])
        print(f"  [+] Energía Exacta (E0): {E_exact:.6f}")

        # 2. Configuración del Modelo
        model = ARSpinViT_Causal(
            hilbert=hi,
            embedding_d=8,
            n_heads=2,
            n_blocks=2,
            n_ffn_layers=1
        )
        
        # Muestreo Directo
        sampler = nk.sampler.ARDirectSampler(hi)
        vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
        optimizer = optax.adam(learning_rate=0.001)
        gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)
        keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

        base_log_name = os.path.join(drive_dir, f"resultado_direct_alpha{alpha}_J{J}")
        log = nk.logging.JsonLog(base_log_name, save_params=False)

        # 3. Calentamiento de JAX (1 iteración)
        print(f"  [+] Calentando compilador JAX (1 época)...")
        gs.run(n_iter=1, out=log, callback=keeper.update)
        jax.block_until_ready(vstate.variables)

        # 4. Entrenamiento Real (499 iteraciones) cronometrado
        print(f"  [+] Entrenando las 499 épocas restantes...")
        start_time = time.time()
        gs.run(n_iter=499, out=log, show_progress=True, callback=keeper.update)
        jax.block_until_ready(vstate.variables)
        
        # Calcular tiempo total equivalente a 500 épocas (sin ruido de compilación)
        time_499 = time.time() - start_time
        exec_time_500 = time_499 * (500 / 499)
        
        # 5. Cargar el mejor estado encontrado
        vstate.parameters = keeper.best_state.parameters
        
        # 6. Cálculo de Métricas Físicas
        print("  [+] Calculando métricas finales y autocorrelación...")
        E_stat = vstate.expect(H)
        E_mean = E_stat.mean.real
        E_var = E_stat.variance.real 
        pearson_dev = float(jnp.sqrt(E_var) / abs(E_mean))

        psi_exact = evecs[:, 0]
        psi_vmc = vstate.to_array(normalize=True)
        overlap = float(jnp.abs(jnp.vdot(psi_exact, psi_vmc))**2)

        # ------------------------------------------------------------------
        # 7. BLOQUE DE AUTOCORRELACIÓN (Integrado y adaptado a Muestreo Directo)
        # ------------------------------------------------------------------
        # Extraemos los estimadores locales. Hacemos .flatten() por si NetKet 
        # devuelve shape (n_chains, n_samples) para evitar errores de índice.
        E_loc_raw = np.array(vstate.local_estimators(H).real)
        E_loc = E_loc_raw[0] if E_loc_raw.ndim > 1 else E_loc_raw

        E_mean_chain = np.mean(E_loc)
        E_var_chain = np.var(E_loc)
        
        t_10_percent = "> Max Lag" 
        max_lag_search = min(200, len(E_loc) - 1) 
        
        for t in range(1, max_lag_search):
            cov_t = np.mean((E_loc[:-t] - E_mean_chain) * (E_loc[t:] - E_mean_chain))
            c_t = cov_t / E_var_chain
            
            if c_t <= 0.1:
                t_10_percent = t
                break
                
        print(f"  [+] Pasos hasta decorrelación del 10%: {t_10_percent}")

        # Títulos y nombres de archivo adaptados a Muestreo Directo
        benchmark_title = f"ARViT Directo (J={J}, alpha={alpha})"
        plot_filename = os.path.join(drive_dir, f"autocorr_direct_alpha{alpha}_J{J}.png")
        
        plot_markov_autocorrelation(
            vstate=vstate, 
            H=H, 
            benchmark_name=benchmark_title, 
            max_lag=100, 
            filename=plot_filename 
        )
        print(f"  [√] Gráfica de autocorrelación guardada en: {os.path.basename(plot_filename)}")
        # ------------------------------------------------------------------

        # 8. Impresión de Resultados
        print("\n>>> RESULTADOS FINALES:")
        print(f"      - Energia VMC       : {E_mean:.6f}")
        print(f"      - Energia Exacta    : {E_exact:.6f}")
        print(f"      - Error Relativo    : {abs((E_mean - E_exact)/E_exact):.8%}")
        print(f"      - Desviacion Pearson: {pearson_dev:.6f}")
        print(f"      - Fidelidad         : {overlap:.6f}")
        print(f"      - Tau (10%)         : {t_10_percent}")
        print(f"      - Tiempo            : {exec_time_500:.1f} s")
        print("-" * 80)
        
if __name__ == "__main__":
    run_direct_sampling_6_points(N=10)