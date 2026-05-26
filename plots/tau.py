import os
import sys
import jax
import netket as nk
import optax

# Ajuste de rutas
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from physics.hamiltonian import get_Hamiltonian
from models.vitB import ARSpinViT_Causal
from physics.utils import BestIterKeeper 
from physics.utils import plot_markov_autocorrelation

def run_metropolis_autocorr_plot():
    print(">>> GENERANDO GRÁFICA DE AUTOCORRELACIÓN: METROPOLIS MCMC")
    print(">>> Punto Crítico: J = 7.0 | alpha = 6.0")
    print("---------------------------------------------------------")
    
    N = 10
    J_val = 7.0
    alpha_val = 6.0
    
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=J_val, alpha=alpha_val, hilbert=hi)

    model = ARSpinViT_Causal(
        hilbert=hi,
        embedding_d=8,
        n_heads=2,
        n_blocks=2,
        n_ffn_layers=1
    )

    # USAMOS METROPOLIS LOCAL
    sampler = nk.sampler.MetropolisLocal(
        hi,
        n_chains=1, # Número de exploradores en paralelo
        sweep_size=1   # Muestras que se dejan pasar entre extracciones
    )
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)
    
    optimizer = optax.adam(learning_rate=0.001)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)
    keeper = BestIterKeeper(Hamiltonian=H, N=N, baseline=1e-6)

    print("Entrenando 500 épocas para recrear el estado crítico (aprox. 5 min)...")
    # No guardamos log en disco porque solo queremos la gráfica al final
    gs.run(n_iter=500, show_progress=True, callback=keeper.update)
    
    print("\nRestaurando la mejor época encontrada...")
    vstate.parameters = keeper.best_state.parameters

    # --- GENERACIÓN DE LA GRÁFICA ---
    print("Calculando y dibujando la autocorrelación...")
    
    # Aseguramos que el directorio plots exista
    out_dir = "/content/TFG_ARViT/plots/"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "autocorr_Metropolis_J7_alpha6.png")
    
    benchmark_title = "Metropolis MCMC (J=7.0, alpha=6.0)"
    
    plot_markov_autocorrelation(
        vstate=vstate, 
        H=H, 
        benchmark_name=benchmark_title, 
        max_lag=100,  # <-- VITAL: Aumentado a 100 para ver la caída completa de un tau=23
        filename=out_file 
    )
    
    print(f"\n[√] ¡Misión cumplida! Gráfica guardada en: {out_file}")
    
    # Copiar a Drive automáticamente por comodidad
    drive_dest = "/content/drive/MyDrive/TFG_ARViT/plots/"
    if os.path.exists("/content/drive/MyDrive/"):
        os.makedirs(drive_dest, exist_ok=True)
        os.system(f"cp {out_file} {drive_dest}")
        print(f"[√] Copia de seguridad guardada en tu Drive.")

if __name__ == "__main__":
    run_metropolis_autocorr_plot()