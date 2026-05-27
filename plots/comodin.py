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
E_calc = vstate.expect(H).mean.real
err_rel = abs((E_calc - E_exact) / E_exact) * 100

psi_vmc = vstate.to_array()
psi_vmc /= jnp.linalg.norm(psi_vmc)
fidelidad = float(jnp.abs(jnp.vdot(psi_vmc, psi_exact))**2)

# Cálculo estadístico manual de Tau (más preciso que el atributo del sampler)
# Extraemos muestras de la última iteración
samples = vstate.samples 
# statistics() de NetKet nos da el tau integrado (tau_int)
stats = statistics(samples)
tau_c = stats.tau_int[0] 

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
# Calculamos la función de autocorrelación sobre las muestras crudas
x = samples[:, 0, :].reshape(-1) # Aplanamos las muestras de la cadena
x = x - np.mean(x)
ac = np.correlate(x, x, mode='full')[len(x)-1:]
ac /= ac[0] # Normalizar

plt.figure(figsize=(8, 4))
plt.plot(ac[:100], color='blue', label='Autocorrelación')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.title(f"Autocorrelación de Muestras (J={J}, $\\alpha$={alpha})")
plt.xlabel("Lag (pasos de Metropolis)")
plt.ylabel("Correlación")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("autocorr_manual.png")
print("[✔] Gráfica 'autocorr_manual.png' guardada.")





