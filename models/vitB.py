import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk

# --- 1. BLOQUE TRANSFORMER CAUSAL ---

class CausalTransformerBlock(nn.Module):
    n_heads: int
    n_ffn_layers: int
    embedding_d: int

    @nn.compact
    def __call__(self, x, mask):
        # Multi-Head Attention pasándole la máscara causal
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.embedding_d,
            out_features=self.embedding_d,
            param_dtype=jnp.float64
        )(x, x, mask=mask)
        
        x = x + attn_out
        x = nn.LayerNorm(param_dtype=jnp.float64)(x)

        # Feed-Forward Network (MLP)
        ffn = x
        for _ in range(self.n_ffn_layers):
            ffn = nn.Dense(features=self.embedding_d * 2, param_dtype=jnp.float64)(ffn)
            ffn = nn.gelu(ffn)
        ffn = nn.Dense(features=self.embedding_d, param_dtype=jnp.float64)(ffn)

        x = x + ffn
        x = nn.LayerNorm(param_dtype=jnp.float64)(x)
        return x

# --- 2. EL MODELO ViT AUTOREGRESIVO

class ARSpinViT_Causal(nk.models.AbstractARNN):
    embedding_d: int = 8
    n_heads: int = 2
    n_blocks: int = 2
    n_ffn_layers: int = 1

    @nn.compact
    def conditionals_log_psi(self, inputs):
        
        batch_size, N = inputs.shape

        # 1. SHIFT CAUSAL: Añadimos 0 al inicio, quitamos el último
        zeros = jnp.zeros((batch_size, 1), dtype=inputs.dtype)
        x = jnp.concatenate([zeros, inputs[:, :-1]], axis=1)
        x = x.astype(jnp.float64)[..., None] 

        # 2. EMBEDDING INICIAL
        x = nn.Dense(features=self.embedding_d, name="embed", param_dtype=jnp.float64)(x)
        
        pos_emb = self.param(
            'pos_embedding', 
            nn.initializers.normal(stddev=0.01), 
            (N, self.embedding_d),
            jnp.float64
        )
        x = x + pos_emb

        # 3. MÁSCARA CAUSAL ESTÁNDAR DE FLAX
        mask = nn.make_causal_mask(jnp.empty((batch_size, N)))

        # 4. PASO POR LOS BLOQUES TRANSFORMER
        for i in range(self.n_blocks):
            x = CausalTransformerBlock(
                n_heads=self.n_heads,
                n_ffn_layers=self.n_ffn_layers,
                embedding_d=self.embedding_d,
                name=f"causal_block_{i}" 
            )(x, mask)

        x = nn.LayerNorm(param_dtype=jnp.float64)(x)

        logits = nn.Dense(
            features=2, 
            name="final_dense",
            kernel_init=nn.initializers.zeros, 
            param_dtype=jnp.float64
        )(x)
        
        # 6. CONVERSIÓN A LOG-AMPLITUDES CUÁNTICAS
        # La red devuelve P(x). La amplitud cuántica es sqrt(P(x)).
        # En el espacio logarítmico: log(sqrt(P)) = 0.5 * log(P)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return jnp.asarray(0.5 * log_probs, dtype=jnp.float64)