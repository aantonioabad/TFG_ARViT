import os
# Configuración crítica para evitar conflictos de memoria
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
import optax
from typing import Sequence, Callable

# ==========================================
# 1. ARQUITECTURA DEL MODELO (VERSIÓN ESTABLE)
# ==========================================

REAL_DTYPE = jnp.float64

class ARSpinViT(nk.models.AbstractARNN):
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    machine_pow: int = 2

    @nn.compact
    def conditionals(self, inputs):
        x = inputs.astype(REAL_DTYPE)
        x = x[..., None]
        
        # Embedding
        x = nn.Dense(features=self.embedding_d, param_dtype=REAL_DTYPE, name="embed")(x)

        # Bloques Transformer
        for i in range(self.n_blocks):
            x_norm = nn.LayerNorm(param_dtype=REAL_DTYPE)(x)
            
            # Atención Causal
            head_dim = self.embedding_d // self.n_heads
            qkv = nn.Dense(self.embedding_d * 3, use_bias=False, param_dtype=REAL_DTYPE)(x_norm)
            q, k, v = jnp.split(qkv, 3, axis=-1)
            
            B, N, _ = x.shape
            shape_heads = (B, N, self.n_heads, head_dim)
            q = q.reshape(shape_heads)
            k = k.reshape(shape_heads)
            v = v.reshape(shape_heads)

            dist = jnp.einsum('bihd,bjhd->bhij', q, k) / jnp.sqrt(head_dim)
            
            # Máscara Causal (CRÍTICO: Solo mirar al pasado)
            mask = jnp.tril(jnp.ones((N, N)))
            mask = mask[None, None, :, :] 
            dist = jnp.where(mask > 0, dist, -1e9)
            
            attn = nn.softmax(dist, axis=-1)
            out_heads = jnp.einsum('bhij,bjhd->bihd', attn, v)
            out_attn = out_heads.reshape((B, N, self.embedding_d))
            
            x = x + nn.Dense(self.embedding_d, param_dtype=REAL_DTYPE)(out_attn)
            
            # MLP (Reducido para estabilidad)
            y = nn.LayerNorm(param_dtype=REAL_DTYPE)(x)
            y = nn.Dense(self.embedding_d * 2, param_dtype=REAL_DTYPE)(y) 
            y = nn.gelu(y)
            y = nn.Dense(self.embedding_d, param_dtype=REAL_DTYPE)(y)
            x = x + y

        x = nn.Dense(self.machine_pow, param_dtype=REAL_DTYPE, name="out")(x)
        # Usamos log_cosh de activation para evitar warnings
        return nk.nn.activation.log_cosh(x)

    def __call__(self, inputs):
        # Sobrescribimos para evitar el error interno de NetKet
        log_conditionals = self.conditionals(inputs)
        ids = (inputs + 1) / 2
        ids = ids.astype(jnp.int32)
        ids = ids[..., None]
        selected_logs = jnp.take_along_axis(log_conditionals, ids, axis=-1)
        return jnp.sum(selected_logs.reshape(inputs.shape), axis=-1)

# ==========================================
# 2. CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================

def run_experiment():
    print(">>> Iniciando experimento ESTABLE (main2.py)...")
    
    # Parámetros físicos
    N = 10
    J = 1.0
    alpha = 3.0
    
    hi = nk.hilbert.Spin(s=0.5, N=N)
    
    # Hamiltoniano
    sx = sum([nk.operator.spin.sigmax(hi, i) for i in range(N)])
    def interaction(i, j):
        d = min(abs(i-j), N-abs(i-j))
        return (J / d**alpha) * nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, j)
    sz_sz = sum([interaction(i, j) for i in range(N) for j in range(i+1, N)])
    H = -sx + sz_sz

    # Energía Exacta
    E_exact = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=False)[0]
    print(f"Energía Exacta: {E_exact:.6f}")

    # --- MODELO LITE (Para evitar overfitting/explosión) ---
    print(">>> Construyendo modelo 'Lite'...")
    model = ARSpinViT(
        hilbert=hi,
        embedding_d=8,   # Reducido de 16 a 8
        n_heads=2,       # Reducido de 4 a 2
        n_blocks=2, 
        n_ffn_layers=1 
    )
    
    sampler = nk.sampler.ARDirectSampler(hi)

    # --- MÁS MUESTRAS ---
    # 2048 muestras > ~1600 parámetros -> ESTABILIDAD
    vstate = nk.vqs.MCState(
        sampler, 
        model, 
        n_samples=2048,  
        n_discard_per_chain=0,
        seed=42
    )

    print(f"Parámetros del modelo: {vstate.n_parameters}")

    # --- OPTIMIZADOR CON FRENO ---
    optimizer = nk.optimizer.Sgd(learning_rate=0.01)
    
    # Diag_shift 0.1 estabiliza la curvatura al principio
    sr = nk.optimizer.SR(diag_shift=0.1)

    gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=sr)

    print(">>> Entrenando (300 iteraciones)...")
    log = nk.logging.RuntimeLog()
    
    gs.run(n_iter=300, out=log, show_progress=True)

    E_final = log["Energy"].Mean[-1]
    error = abs((E_final - E_exact) / E_exact)
    print(f"\n>>> RESULTADO FINAL:")
    print(f"Energía obtenida: {E_final:.6f}")
    print(f"Error relativo: {error:.2%}")

if __name__ == "__main__":
    run_experiment()