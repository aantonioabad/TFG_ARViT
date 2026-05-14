import os
import sys
import time
import jax
import jax.numpy as jnp
import netket as nk
import optax
import scipy.sparse.linalg

# Ajuste de rutas para importar tus módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Importamos tu función, tu modelo y las utilidades...
from physics.hamiltonian import get_Hamiltonian
from models.vitB import ARSpinViT_Causal
from physics.utils import BestIterKeeper, plot_markov_autocorrelation

def run_long_range_sweep(N=10, alpha=6.0, J_values=None):
    if J_values is None:
        J_values = [-4.0, -3.0, 1.0, 3.0, 7.0]

    print(f"\n{'='*70}")
    print(f" INICIANDO BARRIDO LONG-RANGE | N={N} | alpha={alpha}")
    print(f" Valores de J a evaluar: {J_values}")
    print(f"{'='*70}\n")

    # Diccionario para guardar las energías exactas y usarlas en las gráficas luego
    exact_energies = {}

    for J in J_values:
        print(f"\n>>> EXPERIMENTO: Fase con J = {J} <<<")
        
        # 1. Definimos el espacio de Hilbert y el Hamiltoniano
        hi = nk.hilbert.Spin(s=1/2, N=N)
        H = get_Hamiltonian(N=N, J=J, alpha=alpha, hilbert=hi)

        # 2. Calculamos la Energía Exacta (Verdad Fundamental)
        H_sparse = H.to_sparse()
        evals, evecs = scipy.sparse.linalg.eigsh(H_sparse, k=1, which="SA")
        E_exact = float(evals[0])
        psi_exact = evecs[:, 0] # Guardamos el vector exacto para la fidelidad
        exact_energies[J] = E_exact
        print(f"  [+] Energía Exacta Teórica: {E_exact:.6f}")

        # 3. Preparamos el ARViT y el Sampler Directo
        model = ARSpinViT_Causal(hilbert=hi)
        sampler = nk.sampler.ARDirectSampler(hi)
        vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
        
        optimizer = optax.adam(learning_rate=0.001)
        gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

        # 4. Keeper para guardar la mejor iteración
        keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

        # 5. Nombre del archivo log dinámico
        log_name = f"resultado_LR_alpha6.0_J{J}"
        log = nk.logging.JsonLog(log_name, save_params=False)

        print("  [+] Entrenando ARViT (500 épocas)...")
        start_time = time.time()
        
        # Ejecutamos el entrenamiento
        gs.run(n_iter=500, out=log, show_progress=True, callback=keeper.update)
        
        jax.block_until_ready(vstate.variables)
        end_time = time.time()

        print("  [+] Restaurando la mejor iteración y calculando métricas...")
        vstate.parameters = keeper.best_state.parameters
        
        # --- CÁLCULO DE MÉTRICAS FINALES ---
        E_stat = vstate.expect(H)
        tau_c = getattr(E_stat, "tau_corr", 0.0)
        
        # Fidelidad (Overlap con el estado exacto)
        psi_vmc = vstate.to_array(normalize=True)
        overlap = float(jnp.abs(jnp.vdot(psi_exact, psi_vmc))**2)

        # 6. Resultados finales de este J
        print(f"  [+] Tiempo de entrenamiento: {end_time - start_time:.2f} s")
        print(f"  [+] Mejor Energía VMC      : {keeper.best_energy:.6f}")
        print(f"  [+] Error Relativo         : {abs((keeper.best_energy - E_exact)/E_exact):.3%}")
        print(f"  [+] Fidelidad (Overlap)    : {overlap:.6f}")
        print(f"  [+] Autocorrelación tau    : {tau_c:.4f}")
        
        # --- GENERAMOS LA GRÁFICA DE AUTOCORRELACIÓN PARA ESTE J ---
        plot_markov_autocorrelation(
            vstate=vstate, 
            H=H, 
            benchmark_name=f"ARViT Long-Range (J = {J})", 
            max_lag=40, 
            filename=f"autocorr_LR_J_{J}.png"
        )
        print("-" * 70)

    # Imprimimos un resumen final para que puedas copiar las energías exactas
    print("\n" + "="*70)
    print(" RESUMEN DE ENERGÍAS EXACTAS (Para tus scripts de gráficas)")
    print("="*70)
    for J, E in exact_energies.items():
        print(f"J = {J:<5} -> E_exacta = {E:.6f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    
    run_long_range_sweep(N=10, alpha=6.0, J_values=[-4.0, -3.0, 1.0, 3.0, 7.0])