import copy
import pathlib
from typing import Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from netket.sampler import MetropolisRule
from netket.utils.struct import dataclass

REAL_DTYPE = jnp.asarray(1.0).dtype


def circulant(
    row: npt.ArrayLike, times: Optional[int] = None
) -> npt.ArrayLike:
    """Build a (full or partial) circulant matrix based on an array.

    Args:
        row: The first row of the matrix.
        times: If not None, the number of rows to generate.

    Returns:
        If `times` is None, a square matrix with all the offset versions of the
        first argument. Otherwise, `times` rows of a circulant matrix.
    """
    row = jnp.asarray(row)

    def scan_arg(carry, _):
        new_carry = jnp.roll(carry, -1)
        return (new_carry, new_carry)

    if times is None:
        nruter = jax.lax.scan(scan_arg, row, row)[1][::-1, :]
    else:
        nruter = jax.lax.scan(scan_arg, row, None, length=times)[1][::-1, :]

    return nruter


class BestIterKeeper:
    """Store the values of a bunch of quantities from the best iteration.

    "Best" is defined in the sense of lowest energy.

    Args:
        Hamiltonian: An array containing the Hamiltonian matrix.
        N: The number of spins in the chain.
        baseline: A lower bound for the V score. If the V score of the best
            iteration falls under this threshold, the process will be stopped
            early.
        filename: Either None or a file to write the best state to.
    """

    def __init__(
        self,
        Hamiltonian: npt.ArrayLike,
        N: int,
        baseline: float,
        filename: Optional[pathlib.Path] = None,
    ):
        self.Hamiltonian = Hamiltonian
        self.N = N
        self.baseline = baseline
        self.filename = filename
        self.vscore = np.inf
        self.best_energy = np.inf
        self.best_state = None

    def update(self, step, log_data, driver):
        """Update the stored quantities if necessary.

        This function is intended to act as a callback for NetKet. Please refer
        to its API documentation for a detailed explanation.
        """
        vstate = driver.state
        energystep = np.real(vstate.expect(self.Hamiltonian).mean)
        var = np.real(getattr(log_data[driver._loss_name], "variance"))
        mean = np.real(getattr(log_data[driver._loss_name], "mean"))
        varstep = self.N * var / mean**2

        if self.best_energy > energystep:
            self.best_energy = energystep
            self.best_state = copy.copy(driver.state)
            self.best_state.parameters = flax.core.copy(
                driver.state.parameters
            )
            self.vscore = varstep

            if self.filename != None:
                with open(self.filename, "wb") as file:
                    file.write(flax.serialization.to_bytes(driver.state))

        return self.vscore > self.baseline


@dataclass
class InvertMagnetization(MetropolisRule):
    """Monte Carlo mutation rule that inverts all the spins.

    Please refer to the NetKet API documentation for a detailed explanation of
    the MetropolisRule interface.
    """

    def transition(rule, sampler, machine, parameters, state, key, σ):
        indxs = jax.random.randint(
            key, shape=(1,), minval=0, maxval=sampler.n_chains
        )
        σp = σ.at[indxs, :].multiply(-1)
        return σp, None
    


def acf_helper(x):
    x_centered = x - np.mean(x)
    norm = np.sum(x_centered**2)
    if norm == 0: return np.zeros(len(x))
    corr = np.correlate(x_centered, x_centered, mode='full')
    return corr[len(corr)//2:] / norm

def plot_markov_autocorrelation(
    vstate, 
    H, 
    benchmark_name: str, 
    max_lag=50, 
    filename="autocorrelacion.png"
):
    """
    Genera una gráfica de autocorrelación temporal profesional y científica
    para una tesis (TFG). Limita colores, aumenta legibilidad y personaliza el título.
    """
    print(f"\nGenerando gráfica profesional para '{benchmark_name}' en: {filename} ...")
    
    # 1. Obtenemos las energías locales
    E_loc = vstate.local_estimators(H).real
    
    # 2. Manejamos Metropolis (2D chains, samples) vs Directo (1D plano)
    if E_loc.ndim > 1:
        cadena = np.array(E_loc[0, :]) # Cogemos la primera cadena
    else:
        cadena = np.array(E_loc)
        
    # 3. Función ACF
    autocorr_values = acf_helper(cadena)
    
    # Limitamos los ejes
    limit = min(max_lag, len(autocorr_values))
    lags = np.arange(limit)
    autocorr_values = autocorr_values[:limit]

    # --- CONFIGURACIÓN DE ESTILO PROFESIONAL/CIENTÍFICO ---
    # Usamos rcParams.update({}) de forma temporal solo para esta figura.
    # Esto asegura tipografía clara y tamaños legibles para impresión TFG.
    
    with plt.rc_context({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.labelpad': 10,
        'axes.titlepad': 20,
        'grid.color': "#DDDDDD", # Gris muy sutil y claro para la cuadrícula
        'grid.alpha': 0.6,
        'axes.spines.top': False, # Moderno/Limpio: quitar borde superior
        'axes.spines.right': False, # Moderno/Limpio: quitar borde derecho
    }):
        # Usamos un color oscuro/negro para línea/puntos (evitamos el azul chillón)
        color_data = "#111111" # Casi negro nítido
        color_zero = "#777777" # Gris medio sutil para la referencia

        plt.figure(figsize=(9, 6), dpi=100) # Un poco más grande para tesis

        # Pintamos datos (línea nítida y puntos pequeños)
        plt.plot(lags, autocorr_values, marker='o', linestyle='-', color=color_data, markersize=4.5, linewidth=1.2)
        
        # Línea de referencia cero: sutil, gris y discontinua
        plt.axhline(0, color=color_zero, linestyle='--', linewidth=1, alpha=0.9)
        
        # Título personalizado incluyendo el nombre del benchmark
        plt.title(f"Decaimiento de Autocorrelación: {benchmark_name}")
        
        # Etiquetas en español con notación matemática (legibles y profesionales)
        plt.xlabel("Distancia en la cadena (Lag $t$, pasos cadena)")
        plt.ylabel("Autocorrelación $C(t)$ (Energía)")
        
        plt.grid(True)
        plt.tight_layout() # Asegura que los márgenes se respeten
        
        # Guardamos en alta calidad (300 dpi es estándar para impresión)
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[ÉXITO] Gráfica guardada como '{filename}'")