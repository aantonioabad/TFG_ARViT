import os
import sys

# Obtenemos la ruta de la carpeta raíz (/content/TFG_ARViT)
# Como estamos en /content/TFG_ARViT/plots, el padre es /content/TFG_ARViT
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Añadimos la raíz al path si no está ya
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# AHORA ya puedes importar physics
from physics.hamiltonian import get_Hamiltonian
import jax
import jax.numpy as jnp
import netket as nk
import optax
import time
import numpy as np
import scipy.sparse.linalg as sps
import matplotlib.pyplot as plt
from netket.stats import statistics

# --- CONFIGURACIÓN ---
N = 10
J = 7.0
alpha = 6.0
n_iter = 500

# 1. Definir Hamiltoniano y Red (Asegúrate de tener tus módulos en el PATH)
hi = nk.hilbert.Spin(s=0.5, N=N)
from physics.hamiltonian import get_Hamiltonian 
H = get_Hamiltonian(N=N, J=J, alpha=alpha, hilbert=hi)

# Calcular Energía Exacta y Estado Exacto para Fidelidad
H_sparse = H.to_sparse()
eigvals, eigvecs = sps.eigsh(H_sparse, k=1, which="SA")
E_exact = eigvals[0]
psi_exact = eigvecs[:, 0]

from models.vitB import ARSpinViT_Causal
model = ARSpinViT_Causal(hilbert=hi, embedding_d=8, n_heads=2, n_blocks=2, n_ffn_layers=1)

# 2. Metropolis con Burn-in
sampler = nk.sampler.MetropolisLocal(hi)
vstate = nk.vqs.MCState(sampler, model, n_samples=2048, n_discard_per_chain=200, seed=42)
logger = nk.logging.RuntimeLog()

# 3. Entrenamiento
optimizer = optax.adam(learning_rate=0.001)
gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

print(f"[*] Iniciando entrenamiento (Burn-in 200) para J={J}, alpha={alpha}...")
start = time.time()
gs.run(n_iter=n_iter, out=logger)
print(f"[+] Entrenamiento completado en {time.time()-start:.2f}s.")

# 4. Cálculo de Métricas Rigurosas
# --- REEMPLAZA EL BLOQUE DE LAS MÉTRICAS POR ESTE ---

# 4. Cálculo de Métricas Finales
E_calc = vstate.expect(H).mean.real
err_rel = abs((E_calc - E_exact) / E_exact) * 100

# Fidelidad
psi_vmc = vstate.to_array()
psi_vmc /= jnp.linalg.norm(psi_vmc)
fidelidad = float(jnp.abs(jnp.vdot(psi_vmc, psi_exact))**2)

# Cálculo manual de Tau (Evita el NotImplemetedError de NetKet)
# raw_samples tiene forma (n_chains, n_samples_per_chain, N_spins)
# Aplanamos para analizar la serie temporal de espines
raw_samples = vstate.samples[0, :, :].reshape(-1) 

def compute_tau_int(x):
    """Calcula el tiempo de autocorrelación integrado de forma robusta."""
    x = x - np.mean(x)
    n = len(x)
    # Función de autocorrelación normalizada
    result = np.correlate(x, x, mode='full')[n-1:]
    result /= result[0]
    
    # Integramos hasta que la correlación sea insignificante
    # (Metodología estándar para determinar pasos independientes en MCMC)
    tau_int = 0.5 + np.sum(result[1:])
    return max(tau_int, 0.0)

tau_c = compute_tau_int(raw_samples)

print("\n" + "="*35)
print("🎯 MÉTRICAS COMPLETAS PARA TFG")
print("="*35)
print(f"Energía Exacta  : {E_exact:.6f}")
print(f"Energía VMC     : {E_calc:.6f}")
print(f"Error Relativo  : {err_rel:.4f} %")
print(f"Fidelidad (F)   : {fidelidad:.6f}")
print(f"Tau_c (int)     : {tau_c:.4f}")
print("="*35)

# 5. Gráfica de Autocorrelación
# Representamos la caída de la correlación para justificar el burn-in/muestreo
x = raw_samples - np.mean(raw_samples)
ac = np.correlate(x, x, mode='full')[len(x)-1:]
ac /= ac[0]

plt.figure(figsize=(8, 4))
plt.plot(ac[:100], color='blue', label='Correlación $\langle s_i(t)s_i(0) \rangle$')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.title(f"Caída de la Autocorrelación (J={J}, $\\alpha$={alpha})")
plt.xlabel("Lag (pasos de Metropolis)")
plt.ylabel("Correlación")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("autocorr_plot.png")
print("[✔] Gráfica 'autocorr_plot.png' guardada.")