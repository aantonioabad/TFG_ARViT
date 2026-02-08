import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

REAL_DTYPE = jnp.float64

class ARSpinViT(nk.models.AbstractARNN):
    """
    Implementación de Vision Transformer Autoregresivo para funciones de onda cuánticas.
    Versión ESTABILIZADA para evitar NaNs.
    """
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    machine_pow: int = 2

    @nn.compact
    def conditionals(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs.astype(REAL_DTYPE)
        x = x[..., None]
        
        # 1. Embedding
        x = nn.Dense(features=self.embedding_d, param_dtype=REAL_DTYPE, name="embed")(x)

        # 2. Bloques Transformer
        for i in range(self.n_blocks):
            x_norm = nn.LayerNorm(param_dtype=REAL_DTYPE)(x)
            
            # Atención
            head_dim = self.embedding_d // self.n_heads
            qkv = nn.Dense(self.embedding_d * 3, use_bias=False, param_dtype=REAL_DTYPE)(x_norm)
            q, k, v = jnp.split(qkv, 3, axis=-1)
            
            B, N, _ = x.shape
            shape_heads = (B, N, self.n_heads, head_dim)
            q = q.reshape(shape_heads)
            k = k.reshape(shape_heads)
            v = v.reshape(shape_heads)

            dist = jnp.einsum('bihd,bjhd->bhij', q, k) / jnp.sqrt(head_dim)
            
            # Máscara Causal
            mask = jnp.tril(jnp.ones((N, N)))
            mask = mask[None, None, :, :] 
            dist = jnp.where(mask > 0, dist, -1e9)
            
            attn = nn.softmax(dist, axis=-1)
            out_heads = jnp.einsum('bhij,bjhd->bihd', attn, v)
            out_attn = out_heads.reshape((B, N, self.embedding_d))
            
            x = x + nn.Dense(self.embedding_d, param_dtype=REAL_DTYPE)(out_attn)
            
            # MLP
            y = nn.LayerNorm(param_dtype=REAL_DTYPE)(x)
            y = nn.Dense(self.embedding_d * 2, param_dtype=REAL_DTYPE)(y) 
            y = nn.gelu(y)
            y = nn.Dense(self.embedding_d, param_dtype=REAL_DTYPE)(y)
            x = x + y

        # 3. Capa de Salida [FIX CRÍTICO: INICIALIZACIÓN SUAVE]
        # Inicializamos los pesos muy cerca de cero (0.01) para que la red empiece "plana".
        # Esto evita que el primer paso del gradiente sea gigante.
        x = nn.Dense(
            self.machine_pow, 
            param_dtype=REAL_DTYPE, 
            kernel_init=nn.initializers.normal(stddev=0.01), 
            name="out"
        )(x)
        
        return nk.nn.activation.log_cosh(x)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # Override manual para evitar bug de NetKet/JAX
        log_conditionals = self.conditionals(inputs)
        ids = (inputs + 1) / 2
        ids = ids.astype(jnp.int32)
        ids = ids[..., None]
        selected_logs = jnp.take_along_axis(log_conditionals, ids, axis=-1)
        return jnp.sum(selected_logs.reshape(inputs.shape), axis=-1)