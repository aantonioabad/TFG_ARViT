import os
import sys
import time
import json
import jax
import jax.numpy as jnp
import netket as nk
import optax
import scipy.sparse.linalg

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from physics.hamiltonian import get_Hamiltonian
from models.ARViT import ARSpinViT_Causal
from physics.utils import BestIterKeeper

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
    print(f"🚀 INICIANDO BENCHMARK: DIRECT SAMPLING + MÉTRICAS COMPLETAS (6 PUNTOS)")
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
        # Evaluamos el Hamiltoniano con el mejor estado para obtener E_VQMC y Varianza
        stats = vstate.expect(H)
        E_vqmc = float(stats.mean.real)
        Var_E = float(stats.variance)
        
        E_rel = abs((E_vqmc - E_exact) / E_exact)
        delta_E = abs(E_vqmc - E_exact)
        v_score = (N * Var_E) / (delta_E**2) if delta_E > 1e-12 else 0.0

        # Cálculo de Fidelidad Cuántica
        psi_exact = evecs[:, 0]  
        psi_arvit = vstate.to_array()  
        psi_arvit = psi_arvit / jnp.linalg.norm(psi_arvit)  
        fidelidad = float(jnp.abs(jnp.vdot(psi_arvit, psi_exact))**2)
        
        # Mostrar resumen de métricas por pantalla
        print(f"  [✔] RESULTADOS FINALES:")
        print(f"      - E_exacta:  {E_exact:.6f}")
        print(f"      - E_VQMC:    {E_vqmc:.6f}")
        print(f"      - Var(E):    {Var_E:.6f}")
        print(f"      - Error Rel: {E_rel:.6e}")
        print(f"      - V-Score:   {v_score:.4f}")
        print(f"      - Fidelidad: {fidelidad:.6f}")
        print(f"      - Tiempo:    {exec_time_500:.1f} s")
        
        del log 
        
        # 7. Inyectar TODAS las métricas en el archivo JSON generado
        log_path = base_log_name + ".log"
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                log_data = json.load(f)
            
            # Guardamos las métricas extra en una nueva clave para fácil acceso
            log_data['Resumen_Metricas'] = {
                'Alpha': alpha,
                'J': J,
                'E_0': E_exact,
                'E_VQMC': E_vqmc,
                'Var_E': Var_E,
                'E_rel': E_rel,
                'V_Score': v_score,
                'Fidelidad': fidelidad,
                'Tiempo_ejecucion_s': round(exec_time_500, 2)
            }
            
            with open(log_path, 'w') as f:
                json.dump(log_data, f, indent=4)
            print(f"  [💾] Métricas inyectadas con éxito en: {os.path.basename(log_path)}")
        
        print("-" * 60)

if __name__ == "__main__":
    run_direct_sampling_6_points(N=10)