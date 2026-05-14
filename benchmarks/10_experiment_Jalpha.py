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

def run_total_rebuild(N=10):
    # Ruta a tu Drive
    drive_dir = "/content/drive/MyDrive/TFG_ARViT/Fase_ J_alpha/"
    os.makedirs(drive_dir, exist_ok=True)

    # Definimos el mapa maestro de experimentos (15 puntos en total)
    experimentos = {
        6.0: [-4.0, -3.0, 1.0, 3.0, 7.0],      # Caso local
        2.5: [-4.0, -2.0, 1.0, 4.75, 7.0],    # Caso medio
        1.0: [-2.0, -0.8, 0.5, 7.0, 12.0]     # Caso largo alcance
    }

    # Diccionario para guardar las energías exactas que calculemos
    exact_energies_summary = {alpha: {} for alpha in experimentos.keys()}

    print(f"\n{'='*80}")
    print(f"🚀 REHACIENDO TODO: 15 SIMULACIONES (3 Alphas x 5 Js)")
    print(f"💾 Destino: {drive_dir}")
    print(f"🧠 Modelo: ARSpinViT_Causal (d=8, heads=2, blocks=2, ffn=1)")
    print(f"{'='*80}\n")

    for alpha in [6.0, 2.5, 1.0]: 
        for J in experimentos[alpha]:
            print(f"\n>>> TRABAJANDO EN: J={J} | alpha={alpha} <<<")
            
            # 1. Hamiltoniano y Energía Exacta (ED)
            hi = nk.hilbert.Spin(s=1/2, N=N)
            H = get_Hamiltonian(N=N, J=J, alpha=alpha, hilbert=hi)
            H_sparse = H.to_sparse()
            evals, evecs = scipy.sparse.linalg.eigsh(H_sparse, k=1, which="SA")
            E_exact = float(evals[0])
            exact_energies_summary[alpha][J] = E_exact
            print(f"  [+] Energía Exacta: {E_exact:.6f}")

            
            model = ARSpinViT_Causal(
                hilbert=hi,
                embedding_d=8,
                n_heads=2,
                n_blocks=2,
                n_ffn_layers=1
            )
            
            # Usamos muestreo directo para entrenar modelos AR
            sampler = nk.sampler.ARDirectSampler(hi)
            
            # Estado variacional
            vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
            
            optimizer = optax.adam(learning_rate=0.001)
            gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)
            keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

            # 3. Nombres de archivos solicitados
            # Log de Netket (ESTO ES LO QUE GENERA LOS .LOG)
            base_log_name = os.path.join(drive_dir, f"resultado_LR_alpha{alpha}_J{J}")
            log = nk.logging.JsonLog(base_log_name, save_params=False)
            
            # Gráfica de Autocorrelación: autocorr_ARViT_J_{J}_alpha_{alpha}.png
            autocorr_img = os.path.join(drive_dir, f"autocorr_ARViT_J_{J}_alpha_{alpha}.png")

            # 4. Ejecución del Entrenamiento
            print(f"  [+] Entrenando 500 épocas...")
            start_time = time.time()
            gs.run(n_iter=500, out=log, show_progress=True, callback=keeper.update)
            jax.block_until_ready(vstate.variables)
            
            # 5. Generar Autocorrelación
            vstate.parameters = keeper.best_state.parameters
            print(f"  [+] Generando Autocorrelación: {os.path.basename(autocorr_img)}")
            
            # Generamos la gráfica asumiendo que utils.py maneja la cadena de Markov
            plot_markov_autocorrelation(
                vstate=vstate, 
                H=H, 
                benchmark_name=f"ARViT (J={J}, alpha={alpha})", 
                max_lag=40, 
                filename=autocorr_img
            )
            
            print(f"  [√] Finalizado en {time.time() - start_time:.1f}s")
            print("-" * 60)

    # 6. RESUMEN FINAL DE ENERGÍAS
    print("\n" + "="*60)
    print(" 📋 COPIA ESTE BLOQUE PARA TU SCRIPT 11 📋")
    print("="*60)
    print("exact_energies = {")
    for a_val in exact_energies_summary:
        print(f"    {a_val}: {{")
        for j_val, e_val in exact_energies_summary[a_val].items():
            print(f"        {j_val}: {e_val:.6f},")
        print("    },")
    print("}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_total_rebuild(N=10)