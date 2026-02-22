import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
import netket as nk

# --- 2. BLOQUE TRANSFORMER CAUSAL ---

class CausalTransformerBlock(nn.Module):
    n_heads: int
    n_ffn_layers: int
    embedding_d: int

    @nn.compact
    def __call__(self, x, mask):
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.embedding_d,
            out_features=self.embedding_d,
            param_dtype=jnp.float64 # Blindaje de precisión
        )(x, x, mask=mask)
        
        x = x + attn_out
        x = nn.LayerNorm(param_dtype=jnp.float64)(x)

        ffn = x
        for _ in range(self.n_ffn_layers):
            ffn = nn.Dense(features=self.embedding_d * 2, param_dtype=jnp.float64)(ffn)
            ffn = nn.gelu(ffn)
        ffn = nn.Dense(features=self.embedding_d, param_dtype=jnp.float64)(ffn)

        x = x + ffn
        x = nn.LayerNorm(param_dtype=jnp.float64)(x)
        return x

# --- 3. EL MODELO ViT AUTOREGRESIVO PERFECTO ---

class ARSpinViT_Causal(nk.models.AbstractARNN):
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int

    @nn.compact
    def conditionals(self, inputs: jt.ArrayLike) -> jt.ArrayLike:
        batch_size, N = inputs.shape

        # 1. SHIFT: Para no ver el futuro
        zeros = jnp.zeros((batch_size, 1), dtype=inputs.dtype)
        x = jnp.concatenate([zeros, inputs[:, :-1]], axis=1)
        x = x.astype(jnp.float64)[..., None] 

        # 2. EMBEDDINGS
        x = nn.Dense(features=self.embedding_d, name="embed", param_dtype=jnp.float64)(x)
        pos_emb = self.param('pos_embedding', 
                             nn.initializers.normal(stddev=0.01), 
                             (N, self.embedding_d),
                             jnp.float64)
        x = x + pos_emb

        # 3. MÁSCARA CAUSAL
        mask = nn.make_causal_mask(jnp.empty((batch_size, N)))

        # 4. BLOQUES TRANSFORMER
        for i in range(self.n_blocks):
            x = CausalTransformerBlock(
                n_heads=self.n_heads,
                n_ffn_layers=self.n_ffn_layers,
                embedding_d=self.embedding_d,
                name=f"causal_block_{i}" 
            )(x, mask)

        x = nn.LayerNorm(param_dtype=jnp.float64)(x)

        # 5. CABEZAL DE SALIDA
        logits = nn.Dense(
            features=2, 
            name="final_dense",
            # LA PIEDRA EN EL ESTANQUE: Pequeño ruido (0.01) para arrancar el aprendizaje
            kernel_init=nn.initializers.normal(stddev=0.01), 
            param_dtype=jnp.float64
        )(x)
        
        # Sincronización perfecta con el Sampler de NetKet
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return 0.5 * log_probs
