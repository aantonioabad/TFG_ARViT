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
    

def plot_markov_autocorrelation(vstate, H, max_lag=50, filename="autocorrelacion.png"):
    """
    Extrae las energías locales y pinta el decaimiento de la autocorrelación
    frente a la distancia en la cadena (Lag t).
    """
    print(f"\nGenerando gráfica de autocorrelación en: {filename} ...")
    
    E_loc = vstate.local_estimators(H).real
    
    # 2. Manejamos Metropolis (2D: n_chains, n_samples) vs Directo (1D plano)
    if E_loc.ndim > 1:
        cadena = np.array(E_loc[0, :])
    else:
        cadena = np.array(E_loc)
        
    # 3. Función matemática de Autocorrelación (ACF)
    def acf(x):
        x_centered = x - np.mean(x)
        norm = np.sum(x_centered**2)
        if norm == 0: 
            return np.zeros(len(x))
        corr = np.correlate(x_centered, x_centered, mode='full')
        return corr[len(corr)//2:] / norm

    autocorr_values = acf(cadena)
    limit = min(max_lag, len(autocorr_values))
    lags = np.arange(limit)
    autocorr_values = autocorr_values[:limit]

    # 5. Pintamos la gráfica
    plt.figure(figsize=(8, 5))
    plt.plot(lags, autocorr_values, marker='o', linestyle='-', color='#1f77b4', markersize=5)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.title("Decaimiento de Autocorrelación entre muestras", fontsize=14)
    plt.xlabel("Distancia en la cadena (Lag $t$)", fontsize=12)
    plt.ylabel("Autocorrelación $C(t)$", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[ÉXITO] Gráfica guardada como {filename}")