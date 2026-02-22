import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
import netket as nk

# --- 1. BLOQUE TRANSFORMER CAUSAL ---

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
            param_dtype=jnp.float64
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

# --- 2. EL MODELO ViT AUTOREGRESIVO ---

class ARSpinViT_Causal(nk.models.AbstractARNN):
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int

    @nn.compact
    def conditionals(self, inputs: jt.ArrayLike) -> jt.ArrayLike:
        batch_size, N = inputs.shape

        
        zeros = jnp.zeros((batch_size, 1), dtype=inputs.dtype)
        x = jnp.concatenate([zeros, inputs[:, :-1]], axis=1)
        x = x.astype(jnp.float64)[..., None] 

        # EMBEDDINGS
        x = nn.Dense(features=self.embedding_d, name="embed", param_dtype=jnp.float64)(x)
        pos_emb = self.param('pos_embedding', 
                             nn.initializers.normal(stddev=0.01), 
                             (N, self.embedding_d),
                             jnp.float64)
        x = x + pos_emb

        # MÁSCARA CAUSAL
        mask = nn.make_causal_mask(jnp.empty((batch_size, N)))

        # BLOQUES TRANSFORMER
        for i in range(self.n_blocks):
            x = CausalTransformerBlock(
                n_heads=self.n_heads,
                n_ffn_layers=self.n_ffn_layers,
                embedding_d=self.embedding_d,
                name=f"causal_block_{i}" 
            )(x, mask)

        x = nn.LayerNorm(param_dtype=jnp.float64)(x)

        # CABEZAL DE SALIDA 
        logits = nn.Dense(
            features=2, 
            name="final_dense",
            kernel_init=nn.initializers.normal(stddev=0.01), 
            param_dtype=jnp.float64
        )(x)
        
        # Obtenemos las probabilidades clásicas y pasamos a Amplitudes Cuánticas
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return jnp.asarray(0.5 * log_probs, dtype=jnp.float64)

    @nn.compact
    def __call__(self, inputs: jt.ArrayLike) -> jt.ArrayLike:
        
        
        # 1. Calculamos todas las log-amplitudes
        log_psi_cond = self.conditionals(inputs)
        
        # 2. Convertimos los espines [-1, 1] a índices [0, 1]
        idx = ((inputs + 1) // 2).astype(jnp.int32)
        idx = jnp.expand_dims(idx, axis=-1)
        
        # 3. Seleccionamos la amplitud del espín real
        selected_log_psi = jnp.take_along_axis(log_psi_cond, idx, axis=-1)
        selected_log_psi = jnp.squeeze(selected_log_psi, axis=-1)
        
        # 4. Regla de la cadena: La log-amplitud total es la suma
        return jnp.sum(selected_log_psi, axis=-1)
