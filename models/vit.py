import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

# Tipo de dato real para precisión
REAL_DTYPE = jnp.float64

class ARSpinViT(nk.models.AbstractARNN):
    """
    Implementación de Vision Transformer Autoregresivo para funciones de onda cuánticas.
    
    Hereda de: netket.models.AbstractARNN
    """
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    machine_pow: int = 2

    @nn.compact
    def conditionals(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula las probabilidades condicionales logarítmicas no normalizadas.
        
        [MODIFICADO]: Uso de @nn.compact en lugar de setup().
        [ORIGINAL]: Solía separar setup() y __call__(). @nn.compact es más moderno en Flax.

        Args:
            inputs (jnp.ndarray): Batch de configuraciones de espines.
                                  Shape: (Batch_Size, N_Spins)
        
        Returns:
            jnp.ndarray: Log-amplitudes condicionales para cada sitio y espín local.
                         Shape: (Batch_Size, N_Spins, Local_Dim=2)
        """
        # Casting a float64 para precisión numérica
        x = inputs.astype(REAL_DTYPE)
        # Añadir dimensión de canal: (Batch, N) -> (Batch, N, 1)
        x = x[..., None]
        
        # 1. Embedding: Proyección lineal de espines a vectores latentes
        x = nn.Dense(features=self.embedding_d, param_dtype=REAL_DTYPE, name="embed")(x)

        # 2. Bloques Transformer
        for i in range(self.n_blocks):
            # Normalización (Pre-Norm)
            x_norm = nn.LayerNorm(param_dtype=REAL_DTYPE)(x)
            
            # --- ATENCIÓN MULTI-CABEZA (MHA) ---
            head_dim = self.embedding_d // self.n_heads
            
            # Proyección Q, K, V combinada
            qkv = nn.Dense(self.embedding_d * 3, use_bias=False, param_dtype=REAL_DTYPE)(x_norm)
            q, k, v = jnp.split(qkv, 3, axis=-1)
            
            # Reshape para separar cabezas: (Batch, N, Heads, Head_Dim)
            B, N, _ = x.shape
            shape_heads = (B, N, self.n_heads, head_dim)
            q = q.reshape(shape_heads)
            k = k.reshape(shape_heads)
            v = v.reshape(shape_heads)

            # Atención Scaled Dot-Product: (Q @ K^T) / sqrt(d)
            # Einsum: 'bihd,bjhd->bhij' (Batch, Heads, i, j)
            dist = jnp.einsum('bihd,bjhd->bhij', q, k) / jnp.sqrt(head_dim)
            
            # --- [MODIFICADO] MÁSCARA CAUSAL MANUAL ---
            # [ORIGINAL]: NetKet suele usar máscaras internas o nn.MultiHeadAttention con mask argument.
            # [CAMBIO]: Implementamos la máscara manualmente con jnp.tril para garantizar
            # que el espín 'i' SOLO vea a los espines 'j <= i'.
            # Esto evita la violación de causalidad autoregresiva.
            mask = jnp.tril(jnp.ones((N, N)))
            mask = mask[None, None, :, :] # Expandir para Batch y Heads
            
            # Aplicar máscara: -1e9 (infinito negativo) donde no se debe mirar
            dist = jnp.where(mask > 0, dist, -1e9)
            
            # Softmax sobre la última dimensión (j)
            attn = nn.softmax(dist, axis=-1)
            
            # Calcular salida ponderada: Attn @ V
            out_heads = jnp.einsum('bhij,bjhd->bihd', attn, v)
            
            # Concatenar cabezas
            out_attn = out_heads.reshape((B, N, self.embedding_d))
            
            # Proyección de salida + Residual Connection
            x = x + nn.Dense(self.embedding_d, param_dtype=REAL_DTYPE)(out_attn)
            
            # --- FEED FORWARD NETWORK (MLP) ---
            y = nn.LayerNorm(param_dtype=REAL_DTYPE)(x)
            # [MODIFICADO]: Reducción de dimensión oculta (x2 en vez de x4) para estabilidad "Lite".
            y = nn.Dense(self.embedding_d * 2, param_dtype=REAL_DTYPE)(y) 
            y = nn.gelu(y)
            y = nn.Dense(self.embedding_d, param_dtype=REAL_DTYPE)(y)
            # Residual Connection
            x = x + y

        # 3. Capa de Salida
        x = nn.Dense(self.machine_pow, param_dtype=REAL_DTYPE, name="out")(x)
        
        # [MODIFICADO]: Uso de nk.nn.activation.log_cosh
        # [ORIGINAL]: nk.nn.log_cosh (versiones antiguas).
        return nk.nn.activation.log_cosh(x)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Calcula la log-amplitud total de la función de onda para un batch.
        log_psi(s) = sum_i log p(s_i | s_{<i})

        [MODIFICADO - CRÍTICO]: Sobrescritura manual de __call__.
        [ORIGINAL]: La clase AbstractARNN implementa esto automáticamente usando cache.
        [RAZÓN]: Conflictos entre versiones recientes de JAX y NetKet causaban un
                 TypeError: NoneType. Esta implementación manual "puentea" el error
                 conectando directamente conditionals() con la suma final.

        Args:
            inputs (jnp.ndarray): Configuraciones de espines (Batch, N).

        Returns:
            jnp.ndarray: Log-amplitud compleja (aunque usamos real) (Batch,).
        """
        # 1. Obtener todas las condicionales para ambos valores de espín (Up/Down)
        log_conditionals = self.conditionals(inputs)
        
        # 2. Seleccionar la probabilidad del espín que realmente ocurrió en la entrada
        # Transformar espines -1/1 a índices 0/1
        ids = (inputs + 1) / 2
        ids = ids.astype(jnp.int32)
        ids = ids[..., None]
        
        # Gather: Para cada sitio, toma el valor correspondiente al espín de entrada
        selected_logs = jnp.take_along_axis(log_conditionals, ids, axis=-1)
        
        # 3. Sumar sobre todos los sitios (Producto de probabilidades en logaritmo)
        return jnp.sum(selected_logs.reshape(inputs.shape), axis=-1)