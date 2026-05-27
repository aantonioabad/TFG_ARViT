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

# Ajuste de rutas para tus módulos
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from physics.hamiltonian import get_Hamiltonian
from models.vitB import ARSpinViT_Causal
from physics.utils import BestIterKeeper

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
    
    # Extraer energía
    e_mean_list = energy_dict.get("Mean", energy_dict.get("mean", energy_dict.get("value", [])))
    energies = [e.get("real", e.get("Mean", e.get("mean", 0.0))) if isinstance(e, dict) else (e.real if isinstance(e, complex) else float(e)) for e in e_mean_list]
    
    # Extraer Autocorrelación (TauCorr)
    tau_list = energy_dict.get("TauCorr", [])
    taus = [float(t) for t in tau_list]
    
    return energies, taus

def run_metropolis_sampling_6_points(N=10):
    # Ruta principal en tu Drive
    drive_dir = "/content/drive/MyDrive/TFG_ARViT/Fase_ J_alpha/"
    os.makedirs(drive_dir, exist_ok=True)

    # Solo los 6 puntos exactos para la comparativa
    experimentos = [
        (6.0, 7.0),
        (6.0, -4.0),
        (2.5, -4.0),
        (2.5, 2.75),
        (1.0, -2.0),
        (1.0, 7.0)
    ]

    exact_energies_summary = {}
    summary_table = []

    print(f"\n{'='*95}")
    print(f"🚀 INICIANDO BENCHMARK: METROPOLIS MCMC + FIDELIDAD + TAU (6 PUNTOS)")
    print(f"💾 Destino: {drive_dir}")
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
        
        if alpha not in exact_energies_summary:
            exact_energies_summary[alpha] = {}
        exact_energies_summary[alpha][J] = E_exact
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
        
        # Estado variacional
        vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
        
        optimizer = optax.adam(learning_rate=0.001)
        gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)
        keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

        # 3. Nombres de archivos
        base_log_name = os.path.join(drive_dir, f"resultado_metropolis_alpha{alpha}_J{J}")
        log = nk.logging.JsonLog(base_log_name, save_params=False)

        # 4. Ejecución del Entrenamiento
        print(f"  [+] Entrenando 500 épocas con Metropolis Sampling...")
        start_time = time.time()
        gs.run(n_iter=500, out=log, show_progress=True, callback=keeper.update)
        jax.block_until_ready(vstate.variables)
        
        del log  # Liberar el archivo log para poder leerlo
        
        # 5. EXTRAER TAU DE LA MEJOR ÉPOCA DESDE EL LOG
        log_path = base_log_name + ".log"
        energies_log, taus_log = extraer_datos_log(log_path)
        
        best_tau = 0.0
        best_iter = 0
        if energies_log and taus_log:
            best_iter = np.argmin(energies_log)
            best_tau = taus_log[best_iter]
            print(f"  [+] Mejor época detectada en log: {best_iter} | Tau: {best_tau:.4f}")
        else:
            print("  [X] Advertencia: No se pudo leer TauCorr del log.")

        # 6. Cargar mejores parámetros
        vstate.parameters = keeper.best_state.parameters
        E_calc = float(vstate.expect(H).mean.real)
        
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
            'Tau': best_tau  # <-- Guardamos el Tau
        })
        
        # Inyectar Fidelidad y Tau en el JSON original por si lo necesitas luego
        if os.path.exists(log_path):
            try:
                # Usamos una lectura simple asumiendo que save_params=False deja un JSON limpio
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                log_data['Best_Fidelity'] = fidelidad
                log_data['Best_Tau'] = best_tau
                log_data['Best_Iter'] = int(best_iter)
                with open(log_path, 'w') as f:
                    json.dump(log_data, f, indent=4)
                print(f"  [💾] Fidelidad y Tau inyectados con éxito en el log.")
            except json.JSONDecodeError:
                print("  [X] No se inyectó en JSON (Formato concatenado de NetKet).")
        
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





