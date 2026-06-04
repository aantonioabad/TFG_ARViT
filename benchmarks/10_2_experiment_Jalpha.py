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
    print(f"🚀 INICIANDO BENCHMARK: METROPOLIS MCMC + FIDELIDAD + DECORRELACIÓN 10% (6 PUNTOS)")
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
        
        del log # Cierre forzado del logger
        
        # 5. CARGAR MEJORES PARÁMETROS Y CALCULAR ENERGÍA
        vstate.parameters = keeper.best_state.parameters
        E_calc = float(vstate.expect(H).mean.real)

        # 6. CÁLCULO DE LA DECORRELACIÓN DISCRETA AL 10%
        print("  [+] Calculando el paso exacto de caída al 10% (C_t <= 0.1)...")
        E_loc = np.array(vstate.local_estimators(H).real)[0]
        
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

        # 7. CALCULAR FIDELIDAD CUÁNTICA EXACTA AL VUELO
        psi_exact = evecs[:, 0]
        psi_arvit = vstate.to_array()
        psi_arvit = psi_arvit / jnp.linalg.norm(psi_arvit)
        
        fidelidad = float(jnp.abs(jnp.vdot(psi_arvit, psi_exact))**2)
        print(f"  [+] Fidelidad Cuántica Calculada: {fidelidad:.6f}")

        err_rel = abs((E_calc - E_exact) / E_exact)
        summary_table.append({
            'J': J,
            'alpha': alpha,
            'E_exact': E_exact,
            'E_calc': E_calc,
            'Fidelidad': fidelidad,
            'Error_Rel': err_rel,
            'Tau_10': t_10_percent
        })
        
        # Intentar inyectar en el JSON
        log_path = base_log_name + ".log"
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                log_data['Best_Fidelity'] = fidelidad
                log_data['Best_Tau_10_percent'] = t_10_percent
                with open(log_path, 'w') as f:
                    json.dump(log_data, f, indent=4)
                print(f"  [💾] Fidelidad y Pasos (10%) inyectados en el log.")
            except Exception:
                pass 
        
        print(f"  [√] Finalizado en {time.time() - start_time:.1f}s")
        print("-" * 60)

    # =====================================================================
    # TABLA RESUMEN FINAL POR CONSOLA--------------
    # =====================================================================
    print(f"\n{'='*95}")
    print("📊 RESUMEN FINAL DEL BENCHMARK (METROPOLIS MCMC)")
    print(f"{'='*95}")
    print(f"{'J':>6} | {'alpha':>6} | {'E Exacta':>12} | {'E Calculada':>12} | {'Error Rel.':>12} | {'Fidelidad':>10} | {'Tau(10%)':>8}")
    print("-" * 95)
    for res in summary_table:
        tau_str = str(res['Tau_10']) if isinstance(res['Tau_10'], int) else res['Tau_10']
        print(f"{res['J']:6.2f} | {res['alpha']:6.2f} | {res['E_exact']:12.6f} | {res['E_calc']:12.6f} | {res['Error_Rel']:12.2e} | {res['Fidelidad']:10.6f} | {tau_str:>8}")
    print(f"{'='*95}\n")

if __name__ == "__main__":
    run_metropolis_sampling_6_points(N=10)