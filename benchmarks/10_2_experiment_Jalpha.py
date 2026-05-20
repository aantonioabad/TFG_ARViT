import os
import sys
import time
import json
import jax
import jax.numpy as jnp
import netket as nk
import optax
import scipy.sparse.linalg

# Ajuste de rutas para tus módulos
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from physics.hamiltonian import get_Hamiltonian
from models.vitB import ARSpinViT_Causal
from physics.utils import BestIterKeeper, plot_markov_autocorrelation

def run_metropolis_benchmark(N=10):
    drive_dir = "/content/drive/MyDrive/TFG_ARViT/Fase_ J_alpha/metropolis/"
    os.makedirs(drive_dir, exist_ok=True)

    experimentos = [
        (6.0, 7.0), (6.0, -4.0),
        (2.5, -4.0), (2.5, 2.75),
        (1.0, -2.0), (1.0, 7.0)
    ]

    exact_energies_summary = {}

    print(f"\n{'='*80}")
    print(f"🚀 INICIANDO BENCHMARK 11.2: METROPOLIS + CALCULO DE FIDELIDAD")
    print(f"{'='*80}\n")

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

        # 2. Configuración del Modelo
        model = ARSpinViT_Causal(hilbert=hi, embedding_d=8, n_heads=2, n_blocks=2, n_ffn_layers=1)
        sampler = nk.sampler.MetropolisLocal(hi)
        vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
        
        optimizer = optax.adam(learning_rate=0.001)
        gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)
        keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

        # 3. Archivos de salida
        base_log_name = os.path.join(drive_dir, f"resultado_metropolis_alpha{alpha}_J{J}")
        log = nk.logging.JsonLog(base_log_name, save_params=False)
        autocorr_img = os.path.join(drive_dir, f"autocorr_metropolis_J_{J}_alpha_{alpha}.png")

        # 4. Entrenamiento
        gs.run(n_iter=500, out=log, show_progress=True, callback=keeper.update)
        jax.block_until_ready(vstate.variables)
        
        # 5. Cargar mejores parámetros y generar Autocorrelación
        vstate.parameters = keeper.best_state.parameters
        plot_markov_autocorrelation(vstate=vstate, H=H, benchmark_name="ARViT", max_lag=40, filename=autocorr_img)
        
       
        psi_exact = evecs[:, 0]  # Estado fundamental exacto de ED
        psi_arvit = vstate.to_array()  # Estado completo de la red neuronal
        psi_arvit = psi_arvit / jnp.linalg.norm(psi_arvit)  # Normalización por seguridad
        
        # Trasponer y multiplicar (producto escalar complejo)
        fidelidad = float(jnp.abs(jnp.vdot(psi_arvit, psi_exact))**2)
        print(f"  [+] Fidelidad Cuántica Calculada: {fidelidad:.6f}")
        
        del log 
        
        # Abrimos el .log recién creado para inyectarle la fidelidad directamente
        log_path = base_log_name + ".log"
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            
            # Guardamos la fidelidad en una clave nueva
            log_data['Best_Fidelity'] = fidelidad
            
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=4)
            print(f"  [💾] Fidelidad inyectada con éxito en el log.")

        print("-" * 60)

if __name__ == "__main__":
    run_metropolis_benchmark(N=10)