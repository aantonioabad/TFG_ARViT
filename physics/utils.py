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

def plot_markov_autocorrelation(vstate, H, benchmark_name, max_lag=100, filename="autocorr.png"):
    """
    Calcula y grafica la autocorrelación de la cadena de Markov generada por el estado variacional `vstate` con respecto al Hamiltoniano `H`.
    """
   
    E_loc = np.array(vstate.local_estimators(H).real)[0]
    
   
    E_mean = np.mean(E_loc)
    E_var = np.var(E_loc)
    
    max_lag = min(max_lag, len(E_loc) - 1)
    lags = np.arange(max_lag)
    autocorr = []
    
    for t in lags:
        if t == 0:
            autocorr.append(1.0)
        else:
            cov_t = np.mean((E_loc[:-t] - E_mean) * (E_loc[t:] - E_mean))
            
            autocorr.append(cov_t / E_var)
            
   
    with plt.rc_context({'font.family': 'serif', 'font.size': 22, 'axes.spines.top': True, 'axes.spines.right': True}):
        plt.figure(figsize=(10, 6), dpi=150)
        
       
        plt.plot(lags, autocorr, marker='o', linestyle='-', color='black', 
                 linewidth=2.5, markersize=8, alpha=0.85)
        
      
        plt.axhline(0, color='gray', linestyle='--', linewidth=2.0, alpha=0.7)
     
        plt.xlabel("Distancia en la cadena", fontsize=24, fontweight='bold')
        plt.ylabel(r"Autocorrelación $C(t)$", fontsize=24, fontweight='bold')
    
        plt.tick_params(axis='both', which='major', labelsize=20)
        
        plt.grid(True, linestyle='-', color='#E5E8E8', linewidth=1.0)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[ÉXITO] Gráfica guardada como '{filename}'")