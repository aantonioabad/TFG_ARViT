import os
import sys
import time
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
    # Ruta a tu Drive
    drive_dir = "/content/drive/MyDrive/TFG_ARViT/Fase_ J_alpha/"
    os.makedirs(drive_dir, exist_ok=True)

    # Definimos la lista exacta de experimentos en el orden solicitado
    experimentos = [
        (6.0, 7.0),
        (6.0, -4.0),
        (2.5, -4.0),
        (2.5, 2.75),
        (1.0, -2.0),
        (1.0, 7.0)
    ]

    # Diccionario para guardar las energías exactas calculadas
    exact_energies_summary = {}

    print(f"\n{'='*80}")
    print(f"🚀 INICIANDO BENCHMARK 11.2: METROPOLIS SAMPLING (6 Puntos Seleccionados)")
    print(f" Destino: {drive_dir}")
    print(f" Modelo: ARSpinViT_Causal (d=8, heads=2, blocks=2, ffn=1)")
    print(f" Sampler: MetropolisLocal (MCMC)")
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
        print(f"  [+] Energía Exacta: {E_exact:.6f}")

        # 2. Configuración del Modelo ARViT con tus hiperparámetros fijados
        model = ARSpinViT_Causal(
            hilbert=hi,
            embedding_d=8,
            n_heads=2,
            n_blocks=2,
            n_ffn_layers=1
        )
        
        # CAMBIO CLAVE: Cambiamos ARDirectSampler por MetropolisLocal
        sampler = nk.sampler.MetropolisLocal(hi)
        
        # Estado variacional con el nuevo sampler de Metropolis
        vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
        
        optimizer = optax.adam(learning_rate=0.001)
        gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)
        keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

        # 3. Nombres de archivos modificados para no sobrescribir el muestreo directo
        base_log_name = os.path.join(drive_dir, f"resultado_metropolis_alpha{alpha}_J{J}")
        log = nk.logging.JsonLog(base_log_name, save_params=False)
        
        autocorr_img = os.path.join(drive_dir, f"autocorr_metropolis_J_{J}_alpha_{alpha}.png")

        # 4. Ejecución del Entrenamiento (500 épocas)
        print(f"  [+] Entrenando 500 épocas con Metropolis...")
        start_time = time.time()
        gs.run(n_iter=500, out=log, show_progress=True, callback=keeper.update)
        jax.block_until_ready(vstate.variables)
        
        # 5. Generar Autocorrelación
        vstate.parameters = keeper.best_state.parameters
        print(f"  [+] Generando Autocorrelación: {os.path.basename(autocorr_img)}")
        
        plot_markov_autocorrelation(
            vstate=vstate, 
            H=H, 
            benchmark_name=f"ARViT Metropolis (J={J}, alpha={alpha})", 
            max_lag=40, 
            filename=autocorr_img
        )
        
        print(f"  [√] Finalizado en {time.time() - start_time:.1f}s")
        print("-" * 60)

    # 6. RESUMEN FINAL DE ENERGÍAS
    print("\n" + "="*60)
    print(" 📋 ENERGÍAS EXACTAS CALCULADAS 📋")
    print("="*60)
    print("exact_energies_metropolis = {")
    for a_val in exact_energies_summary:
        print(f"    {a_val}: {{")
        for j_val, e_val in exact_energies_summary[a_val].items():
            print(f"        {j_val}: {e_val:.6f},")
        print("    },")
    print("}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_metropolis_benchmark(N=10)