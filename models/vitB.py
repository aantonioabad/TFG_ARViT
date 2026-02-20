from typing import Callable, Sequence, Any
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
import netket as nk
import numpy as np

# --- 1. UTILIDADES Y CONSTANTES ---
REAL_DTYPE = jnp.float64

# --- 2. MÓDULOS REUTILIZABLES (Building Blocks) ---

class MultiLayerPerceptron(nn.Module):
    """
    Módulo Flax para un MLP con normalización LayerNorm opcional.
    Igual al estilo del ejemplo proporcionado.
    """
    layer_widths: Sequence[int]
    activation_function: Callable = nn.gelu  # Usamos GELU por defecto para ViT
    kernel_init: Callable = nn.initializers.lecun_normal()
    use_bias: bool = True

    @nn.compact
    def __call__(self, x) -> jt.ArrayLike:
        for w in self.layer_widths:
            # Si la anchura es 1, no usamos LayerNorm para no destruir la señal
            if w == 1:
                normalizer = lambda x: x
            else:
                normalizer = nn.LayerNorm(param_dtype=REAL_DTYPE)
            
            x = normalizer(
                nn.Dense(
                    w,
                    use_bias=self.use_bias,
                    kernel_init=self.kernel_init,
                    param_dtype=REAL_DTYPE,
                )(x)
            )
            
            x = self.activation_function(x)
        return x


class CausalSelfAttention(nn.Module):
    """
    Bloque de Atención Multi-Cabeza con Máscara Causal.
    Fundamental para asegurar la propiedad autoregresiva.
    """
    n_heads: int
    head_size: int
    kernel_init: Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x) -> jt.ArrayLike:
        B, N, _ = x.shape
        embedding_d = self.n_heads * self.head_size

        # 1. Proyecciones Q, K, V
        qkv = nn.Dense(
            embedding_d * 3, 
            use_bias=False, 
            kernel_init=self.kernel_init, 
            param_dtype=REAL_DTYPE
        )(x)
        
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape para separar cabezas: (Batch, N, Heads, Head_Dim)
        shape_heads = (B, N, self.n_heads, self.head_size)
        q = q.reshape(shape_heads)
        k = k.reshape(shape_heads)
        v = v.reshape(shape_heads)

        # 2. Atención Scaled Dot-Product
        # Score: (Batch, Heads, i, j)
        dist = jnp.einsum('bihd,bjhd->bhij', q, k) / jnp.sqrt(self.head_size)
        
        # 3. MÁSCARA CAUSAL (CRÍTICO)
        # El espín 'i' solo puede atender a 'j <= i'
        mask = jnp.tril(jnp.ones((N, N)))
        mask = mask[None, None, :, :] # Expandir dims
        
        # Aplicar máscara (-inf donde no se debe mirar)
        dist = jnp.where(mask > 0, dist, -jnp.inf)
        
        # Softmax
        attn = nn.softmax(dist, axis=-1)
        
        # 4. Salida ponderada
        out_heads = jnp.einsum('bhij,bjhd->bihd', attn, v)
        
        # Concatenar cabezas
        return out_heads.reshape((B, N, embedding_d))


class TransformerBlock(nn.Module):
    """
    Bloque fundamental del Transformer:
    Input -> LayerNorm -> Atención Causal -> Residual -> LayerNorm -> MLP -> Residual
    """
    n_heads: int
    n_ffn_layers: int
    embedding_d: int  # Dimensión total del embedding

    @nn.compact
    def __call__(self, x) -> jt.ArrayLike:
        # Validación de dimensiones
        if self.embedding_d % self.n_heads != 0:
             raise ValueError("Embedding dimension must be divisible by n_heads")
        
        head_size = self.embedding_d // self.n_heads

        # --- Sub-bloque 1: Atención ---
        x_norm1 = nn.LayerNorm(param_dtype=REAL_DTYPE)(x)
        attn_out = CausalSelfAttention(
            n_heads=self.n_heads, 
            head_size=head_size
        )(x_norm1)
        
        x = x + nn.Dense(self.embedding_d, param_dtype=REAL_DTYPE)(attn_out)

        # --- Sub-bloque 2: MLP (Feed Forward) ---
        
        mlp_widths = [self.embedding_d * 2] * self.n_ffn_layers + [self.embedding_d]
        
        x_norm2 = nn.LayerNorm(param_dtype=REAL_DTYPE)(x)
        
        
        # Implementación manual del MLP típico de Transformer para control exacto:
        y = nn.Dense(self.embedding_d * 2, param_dtype=REAL_DTYPE)(x_norm2)
        y = nn.gelu(y)
        y = nn.Dense(self.embedding_d, param_dtype=REAL_DTYPE)(y)
        
        x = x + y # Residual
        
        return x


# --- 3. CLASE BASE COMÚN PARA EL MODELO ---

class ARSpinViTBase(nk.models.AbstractARNN):
    """
    Clase base que define la arquitectura (conditionals).
    No define __call__, eso se deja a las subclases.
    """
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    machine_pow: int = 2
    
    @nn.compact
    def conditionals(self, inputs: jt.ArrayLike) -> jt.ArrayLike:
        """
        Calcula las log-probabilidades condicionales no normalizadas.
        """
        # Preprocesamiento de inputs
        x = inputs.astype(REAL_DTYPE)
        x = x[..., None] # (Batch, N, 1)

        # 1. Embedding Inicial (Valor del espín)
        x = nn.Dense(features=self.embedding_d, param_dtype=REAL_DTYPE, name="embed")(x)

        # --- NUEVO: POSITIONAL EMBEDDING (AÑADIR ESTO) ---
        # Esto le dice a la red: "Tú eres el espín 0, tú el 1..."
        batch_size, N, _ = x.shape
        # Creamos una matriz de parámetros aprendibles (N, embedding_d)
        pos_emb = self.param('pos_embedding', 
                             nn.initializers.normal(stddev=0.02), 
                             (N, self.embedding_d),
                             REAL_DTYPE)
        
        # Se suma al input (Broadcasting automático sobre el Batch)
        x = x + pos_emb
        # -------------------------------------------------

        # 2. Bloques Transformer
        for _ in range(self.n_blocks):
            x = TransformerBlock(
                n_heads=self.n_heads,
                n_ffn_layers=self.n_ffn_layers,
                embedding_d=self.embedding_d
            )(x)

        x = nn.LayerNorm()(x)
        
        # 3. Cabezal de Salida
        x = nn.Dense(
            #self.machine_pow,
            features=2, 
            param_dtype=REAL_DTYPE,
            kernel_init=nn.initializers.normal(stddev=0.01),
            name="out"
        )(x)
        
        #return nk.nn.activation.log_cosh(x)
        return 0.5 * jax.nn.log_softmax(x, axis=-1)
       


# --- 4. VERSIÓN 1: ESTÁNDAR (Call implícito de NetKet) ---

class ARSpinViT_Standard(ARSpinViTBase):
    """
    Versión estándar que confía en la implementación de __call__ de AbstractARNN.
    
    """
    pass


# --- 5. VERSIÓN 2: MANUAL (Tu solución Override) ---

class ARSpinViT_Manual(ARSpinViTBase):
    """
    Versión robusta con __call__ implementado manualmente.
    Soluciona el error 'TypeError: NoneType' en ciertas versiones de JAX/NetKet.
    """
    """
    def __call__(self, inputs: jt.ArrayLike) -> jt.ArrayLike:
        # 1. Obtener condicionales (Batch, N, 2)
        log_conditionals = self.conditionals(inputs)
        
        # 2. Seleccionar la probabilidad del espín real
        # Map spin -1 -> index 0, spin 1 -> index 1
        ids = (inputs + 1) / 2
        ids = ids.astype(jnp.int32)
        ids = ids[..., None]
        # ids= (inputs+1)//2
        
        # Gather
        selected_logs = jnp.take_along_axis(log_conditionals, ids, axis=-1)
        
        # 3. Sumar log-probs sobre todos los sitios (Batch,)
        return jnp.sum(selected_logs.reshape(inputs.shape), axis=-1)
        """
    @nn.compact
    def __call__(self, inputs: jt.ArrayLike) -> jt.ArrayLike:
        import jax.numpy as jnp
        
        # 1. Obtenemos las log-amplitudes correctas llamando a tu nuevo conditionals
        log_conds = self.conditionals(inputs) # Forma: (Batch, N, 2)
        
        # 2. Convertimos los espines reales [-1, 1] a índices de array [0, 1]
        indices = ((inputs + 1) / 2).astype(jnp.int32)
        
        # 3. Escogemos la probabilidad exacta que corresponde al espín que tenemos
        log_psi_n = jnp.take_along_axis(log_conds, jnp.expand_dims(indices, -1), axis=-1)
        log_psi_n = jnp.squeeze(log_psi_n, axis=-1) # Forma: (Batch, N)
        
        # 4. Regla de la cadena: P(total) = P(1)*P(2|1)... -> En logaritmos es una SUMA
        return jnp.sum(log_psi_n, axis=-1)
    
class CausalTransformerBlock(nn.Module):
    n_heads: int
    n_ffn_layers: int
    embedding_d: int

    @nn.compact
    def __call__(self, x, mask):
        # 1. ATENCIÓN CAUSAL (Aquí le pasamos la máscara)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.embedding_d,
            out_features=self.embedding_d
        )(x, x, mask=mask) # <-- El bloqueo del futuro
        
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

class ARSpinViT_Causal(nk.models.AbstractARNN):
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int

    @nn.compact
    def conditionals(self, inputs: jt.ArrayLike) -> jt.ArrayLike:
        batch_size, N = inputs.shape

        # --- 1. EL DESPLAZAMIENTO (SHIFT) ---
        # Metemos un '0' al principio y quitamos el último espín.
        zeros = jnp.zeros((batch_size, 1), dtype=inputs.dtype)
        shifted_inputs = jnp.concatenate([zeros, inputs[:, :-1]], axis=1)
        
        # Pasamos a (Batch, N, 1)
        x = shifted_inputs.astype(jnp.float32)[..., None] 

        # --- 2. EMBEDDINGS ---
        x = nn.Dense(features=self.embedding_d, name="embed")(x)
        pos_emb = self.param('pos_embedding', 
                             nn.initializers.normal(stddev=0.02), 
                             (N, self.embedding_d))
        x = x + pos_emb

        # --- 3. LA MÁSCARA CAUSAL ---
        # Crea una matriz triangular para tapar el futuro en JAX
        mask = nn.make_causal_mask(jnp.empty((batch_size, N)))

        # --- 4. BLOQUES TRANSFORMER CAUSALES ---
        for _ in range(self.n_blocks):
            x = CausalTransformerBlock(
                n_heads=self.n_heads,
                n_ffn_layers=self.n_ffn_layers,
                embedding_d=self.embedding_d
            )(x, mask) # Pasamos la máscara a cada bloque

        x = nn.LayerNorm()(x)

        # --- 5. CABEZAL DE SALIDA ---
        x = nn.Dense(
            features=2, # Probabilidades up/down
            kernel_init=nn.initializers.normal(stddev=0.01),
            name="out"
        )(x)
        
        # Devolvemos los logits crudos, el __call__ se encarga del resto
        return x

    def __call__(self, inputs: jt.ArrayLike) -> jt.ArrayLike:
        # Sincronizamos el cerebro (Evaluación vs Generación)
        logits = self.conditionals(inputs)
        
        # Aplicamos la normalización física obligatoria
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        
        # Seleccionamos la probabilidad del espín real que tenemos
        ids = ((inputs + 1) / 2).astype(jnp.int32)[..., None]
        selected_logs = jnp.take_along_axis(log_probs, ids, axis=-1).squeeze(-1)
        
        # Sumamos por la regla de la cadena y pasamos a Amplitud Cuántica (0.5)
        return 0.5 * jnp.sum(selected_logs, axis=-1)