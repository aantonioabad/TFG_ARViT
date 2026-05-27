import jax
import jax.numpy as jnp
import netket as nk
import optax
import time
import numpy as np
import scipy.sparse.linalg as sps
import matplotlib.pyplot as plt

# Configuración del sistema
N = 10
J = 7.0
alpha = 6.0
n_iter = 500

# 1. Definir Hamiltoniano y Red
hi = nk.hilbert.Spin(s=0.5, N=N)
from physics.hamiltonian import get_Hamiltonian 
H = get_Hamiltonian(N=N, J=J, alpha=alpha, hilbert=hi)

# Calcular Energía Exacta (ED) para el Error Relativo y Fidelidad
H_sparse = H.to_sparse()
eigvals, eigvecs = sps.eigsh(H_sparse, k=1, which="SA")
E_exact = eigvals[0]
psi_exact = eigvecs[:, 0]

from models.vitB import ARSpinViT_Causal
model = ARSpinViT_Causal(hilbert=hi, embedding_d=8, n_heads=2, n_blocks=2, n_ffn_layers=1)

# 2. Metropolis con Burn-in
sampler = nk.sampler.MetropolisLocal(
        hi,
        n_chains=1, # Número de exploradores en paralelo
        sweep_size=1   # Muestras que se dejan pasar entre extracciones
    )

vstate = nk.vqs.MCState(sampler, model, n_samples=2048, n_discard_per_chain=200, seed=42)
logger = nk.logging.RuntimeLog()

# 3. Entrenamiento
optimizer = optax.adam(learning_rate=0.001)
gs = nk.driver.VMC_SR(H, optimizer, variational_state=vstate, diag_shift=0.1)

print(f"[*] Iniciando entrenamiento (J={J}, alpha={alpha})...")
gs.run(n_iter=n_iter, out=logger)

# 4. Cálculo de Métricas Finales
E_calc = vstate.expect(H).mean.real
err_rel = abs((E_calc - E_exact) / E_exact) * 100

# Fidelidad
psi_vmc = vstate.to_array()
psi_vmc /= jnp.linalg.norm(psi_vmc)
fidelidad = float(jnp.abs(jnp.vdot(psi_vmc, psi_exact))**2)

# Autocorrelación (Tau)
tau_c = vstate.sampler_state.tau_corr if hasattr(vstate.sampler_state, 'tau_corr') else 0.0

print("\n" + "="*30)
print("🎯 MÉTRICAS FINALES")
print("="*30)
print(f"Energía Exacta  : {E_exact:.6f}")
print(f"Energía VMC     : {E_calc:.6f}")
print(f"Error Relativo  : {err_rel:.4f} %")
print(f"Fidelidad       : {fidelidad:.6f}")
print(f"Tau_c           : {tau_c:.4f}")

# 5. Gráfica de Autocorrelación (Extraída del logger)
tau_history = logger.data['Energy']['TauCorr']
plt.figure(figsize=(8, 4))
plt.plot(tau_history, color='red', label=r'Autocorrelación $\tau_c$')
plt.axhline(tau_c, linestyle='--', color='black', alpha=0.5, label='Último valor')
plt.xlabel("Épocas")
plt.ylabel(r"$\tau_c$")
plt.title("Evolución de la Autocorrelación")
plt.legend()
plt.grid(True)
plt.savefig("autocorr_plot.png")
print("\n[✔] Gráfica guardada como 'autocorr_plot.png'")





