from typing import Callable, Sequence, Any
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
import netket as nk

# --- 1. UTILIDADES Y CONSTANTES ---
REAL_DTYPE = jnp.float64

# --- 2. BLOQUE TRANSFORMER CAUSAL ---

class CausalTransformerBlock(nn.Module):
    """
    Bloque Transformer con soporte para Máscara Causal.
    """
    n_heads: int
    n_ffn_layers: int
    embedding_d: int

    @nn.compact
    def __call__(self, x, mask):
        # 1. ATENCIÓN CAUSAL
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.embedding_d,
            out_features=self.embedding_d
        )(x, x, mask=mask)
        
        x = x + attn_out
        x = nn.LayerNorm()(x)

        # 2. RED FEED-FORWARD
        ffn = x
        for _ in range(self.n_ffn_layers):
            ffn = nn.Dense(features=self.embedding_d * 2)(ffn)
            ffn = nn.gelu(ffn)
        ffn = nn.Dense(features=self.embedding_d)(ffn)

        x = x + ffn
        x = nn.LayerNorm()(x)
        return x

# --- 3. EL MODELO ViT AUTOREGRESIVO PERFECTO ---

class ARSpinViT_Causal(nk.models.AbstractARNN):
    """
    Vision Transformer adaptado para sistemas de espines de forma Autoregresiva.
    Implementa Shift de entrada y Máscara Causal estricta.
    """
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int

    @nn.compact
    def conditionals(self, inputs: jt.ArrayLike) -> jt.ArrayLike:
        batch_size, N = inputs.shape

        # --- 1. EL DESPLAZAMIENTO (SHIFT) ---
        zeros = jnp.zeros((batch_size, 1), dtype=inputs.dtype)
        shifted_inputs = jnp.concatenate([zeros, inputs[:, :-1]], axis=1)
        x = shifted_inputs.astype(jnp.float32)[..., None] 

        # --- 2. EMBEDDINGS ---
        x = nn.Dense(features=self.embedding_d, name="embed")(x)
        pos_emb = self.param('pos_embedding', 
                             nn.initializers.normal(stddev=0.02), 
                             (N, self.embedding_d))
        x = x + pos_emb

        # --- 3. LA MÁSCARA CAUSAL ---
        mask = nn.make_causal_mask(jnp.empty((batch_size, N)))

        # --- 4. BLOQUES TRANSFORMER ---
        for _ in range(self.n_blocks):
            x = CausalTransformerBlock(
                n_heads=self.n_heads,
                n_ffn_layers=self.n_ffn_layers,
                embedding_d=self.embedding_d
            )(x, mask)

        x = nn.LayerNorm()(x)

        # --- 5. CABEZAL DE SALIDA ---
        x = nn.Dense(
            features=2,
            kernel_init=nn.initializers.normal(stddev=0.01),
            name="out"
        )(x)
        
        return x

    def __call__(self, inputs: jt.ArrayLike) -> jt.ArrayLike:
        logits = self.conditionals(inputs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        ids = ((inputs + 1) / 2).astype(jnp.int32)[..., None]
        selected_logs = jnp.take_along_axis(log_probs, ids, axis=-1).squeeze(-1)
        
        return 0.5 * jnp.sum(selected_logs, axis=-1)