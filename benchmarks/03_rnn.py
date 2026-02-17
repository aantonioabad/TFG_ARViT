import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# Obtiene la ruta del directorio padre (.../TFG_ARViT)
parent_dir = os.path.dirname(current_dir)
# Añade el padre al sistema para poder importar 'physics'
sys.path.append(parent_dir)

# Configuración de JAX para estabilidad
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"


import netket as nk
import optax
import time
from physics.hamiltonian import get_Hamiltonian

def run_rnn():
    print(">>> BENCHMARK 03: RED RECURRENTE (LSTM + Direct Sampling)")
    print("---------------------------------------------------------")
    
    # --- 3. SISTEMA FÍSICO ---
    # Usamos exactamente el mismo sistema que en tu Transformer
    N = 10
    hi = nk.hilbert.Spin(s=0.5, N=N)
    H = get_Hamiltonian(N, J=1.0, alpha=3.0, hilbert=hi)

    # Intentamos cargar la energía exacta para calcular el error en vivo
    try:
        with open("benchmark_exact.txt", "r") as f:
            E_exact = float(f.read())
    except:
        E_exact = None

    # --- 4. EL MODELO (Rival) ---
    # Usamos una LSTM (Long Short-Term Memory)..
    # - layers=2: Profundidad de la red.
    # - features=8: Tamaño de la "memoria" interna (equivalente a embedding_d).
    model = nk.models.LSTMNet(
        layers=2,
        features=8,
        param_dtype=float
    )

    # --- 5. EL SAMPLER ---
    # Las RNNs son autoregresivas por naturaleza (leen 1->2->3...).
    # Por tanto, podemos usar Sampleo Directo (sin autocorrelación).
    sampler = nk.sampler.ARDirectSampler(hi)

    # --- 6. ESTADO VARIACIONAL ---
    # n_samples=2048: Igual que en tu experimento final.
    vstate = nk.vqs.MCState(sampler, model, n_samples=2048, seed=42)

    # --- 7. OPTIMIZADOR ---
    # Usamos Adam, que suele funcionar mejor que SGD para redes recurrentes.
    optimizer = optax.adam(learning_rate=0.01)
    
    # Usamos VMC_SR con diag_shift=0.1 (Modo estable)
    gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

    # --- 8. ENTRENAMIENTO ---
    print("Entrenando LSTM...")
    start_time = time.time()
    
    log = nk.logging.JsonLog("resultado_benchmark_03", save_params=False)
    gs.run(n_iter=300, out=log, show_progress=True)
    
    end_time = time.time()
    
    # --- 9. RESULTADOS ---
    E_final = log["Energy"].Mean[-1]
    
    print(f"\n>>> RESULTADOS RNN (LSTM):")
    print(f"Energía RNN   : {E_final:.6f}")
    
    if E_exact:
        print(f"Energía Exacta: {E_exact:.6f}")
        err = abs((E_final - E_exact)/E_exact)
        print(f"Error Relativo: {err:.2%}")
        
    print(f"Tiempo Total  : {end_time - start_time:.2f} s")

if __name__ == "__main__":
    run_rnn()