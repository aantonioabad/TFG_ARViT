from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.typing as jt
import netket as nk
import numpy.typing as npt

from physics.utils import REAL_DTYPE, circulant

class MultiLayerPerceptron(nn.Module):
    """Flax module for a Multi-layer perceptron architecture with normalization."""
    layer_widths: Sequence[int]
    activation_function: Callable = nn.swish
    kernel_init: Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x) -> jt.ArrayLike:
        for w in self.layer_widths:
            if w == 1:
                normalizer = lambda x: x
            else:
                normalizer = nn.LayerNorm(param_dtype=REAL_DTYPE)
            x = self.activation_function(
                normalizer(
                    nn.Dense(
                        w,
                        kernel_init=self.kernel_init,
                        param_dtype=REAL_DTYPE,
                    )(x)
                )
            )
        return x


class AffinityPosWeight(nn.Module):
    "Flax module that multiplies by a circulant matrix."

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        weight_row = self.param(
            "alpha_delta",
            nn.initializers.truncated_normal(
                stddev=jnp.sqrt(1.0 / x.shape[-2])
            ),
            (x.shape[-2],),
            REAL_DTYPE,
        )

        weight = circulant(weight_row)

        return weight @ x


class PositionalHead(nn.Module):
    """Flax module that implements to a single head of linearized attention."""
    head_size: int

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        value = nn.Dense(
            self.head_size, use_bias=False, param_dtype=REAL_DTYPE
        )
        aff = AffinityPosWeight()
        return aff(value(x))


class MultiHeadPositionalAttention(nn.Module):
    """Flax module implementing a multi-head block."""
    n_heads: int
    head_size: int

    @nn.compact
    def __call__(self, x: jt.ArrayLike) -> jt.ArrayLike:
        heads = [PositionalHead(self.head_size) for _ in range(self.n_heads)]
        return jnp.concatenate([h(x) for h in heads], axis=-1)


class CoreBlock(nn.Module):
    """Flax module implementing the core block of the ViT."""
    n_heads: int
    n_ffn_layers: int

    @nn.compact
    def __call__(self, x) -> jt.ArrayLike:
        embedding_d = x.shape[-1]
        if embedding_d % self.n_heads != 0:
            raise ValueError(
                "The number of heads must divide the embedding dimensions"
            )
        head_size = embedding_d // self.n_heads
        sa = MultiHeadPositionalAttention(self.n_heads, head_size)
        x += sa(nn.LayerNorm(param_dtype=REAL_DTYPE)(x))
        ffn = MultiLayerPerceptron(
            [
                embedding_d,
            ]
            * self.n_ffn_layers
        )
        return nk.nn.log_cosh(ffn(x) + x)


class RealSpinViT(nn.Module):
    """Flax module that implements a real-valued ViT architecture."""
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Sequence[int]

    @nn.compact
    def __call__(self, x):
        embedding = nn.Dense(self.embedding_d, param_dtype=REAL_DTYPE)

        x = embedding(x)

        blocks = [
            CoreBlock(self.n_heads, self.n_ffn_layers)
            for _ in range(self.n_blocks)
        ]

        for cb in blocks:
            x = cb(x)

        
        x = x.sum(axis=0)

        postprocessor = MultiLayerPerceptron(self.final_architecture)
        x = postprocessor(x)

        
        return nn.Dense(1, param_dtype=REAL_DTYPE)(x).squeeze()


class SpinViTWorker(nn.Module):
    """Flax module that implements a real- or complex-valued ViT architecture."""
    token_size: int
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Sequence[int]
    is_complex: bool

    @nn.compact
    def __call__(self, x):
        real_part = RealSpinViT(
            self.embedding_d,
            self.n_heads,
            self.n_blocks,
            self.n_ffn_layers,
            self.final_architecture,
        )(x)

        if self.is_complex:
            imag_part = RealSpinViT(
                self.embedding_d,
                self.n_heads,
                self.n_blocks,
                self.n_ffn_layers,
                self.final_architecture,
            )(x)

            return real_part + 1.0j * imag_part

        return real_part


class SpinViT(nn.Module):
    """Flax module wrapping `SpinViTWorker` and enforcing translation invariance."""
    token_size: int
    embedding_d: int
    n_heads: int
    n_blocks: int
    n_ffn_layers: int
    final_architecture: Sequence[int]
    is_complex: bool

    @nn.compact
    def __call__(self, x):
        worker = SpinViTWorker(
            self.token_size,
            self.embedding_d,
            self.n_heads,
            self.n_blocks,
            self.n_ffn_layers,
            self.final_architecture,
            self.is_complex,
        )
        
        
        if x.ndim == 1:
            x_in = x[..., None]
        else:
            x_in = x

        circulant_x = circulant(x_in, self.token_size)
        
        
        if circulant_x.ndim == 2:
            circulant_x = circulant_x[..., None]

        return jax.vmap(worker, in_axes=0)(circulant_x).mean(axis=0)


class BatchedSpinViT(nn.Module):
    "Batched version of SpinViT, accepting several spin configurations at once."
    token_size: int = 1
    embedding_d: int = 8
    n_heads: int = 2
    n_blocks: int = 2
    n_ffn_layers: int = 1
    final_architecture: Sequence[int] = (8, 4)
    is_complex: bool = False

    @nn.compact
    def __call__(self, batched_x):
        worker = SpinViT(
            self.token_size,
            self.embedding_d,
            self.n_heads,
            self.n_blocks,
            self.n_ffn_layers,
            self.final_architecture,
            self.is_complex,
        )
        return jax.vmap(worker, in_axes=0)(batched_x)