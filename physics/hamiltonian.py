import netket as nk
from netket.operator import LocalOperator
from netket.hilbert import AbstractHilbert

def get_Hamiltonian(N: int, J: float, alpha: float, hilbert: AbstractHilbert) -> LocalOperator:
    """
    Construye el Hamiltoniano de Ising de largo alcance.
    
    Args:
        N: Número de espines.
        J: Intensidad de interacción.
        alpha: Decaimiento de la interacción.
        hilbert: El espacio de Hilbert (objeto Spin de NetKet).
    """
    
    # 1. Definimos la intensidad del campo transversal (h_x)
    h_x = 1.0
    
    # 2. Término de campo transversal: - h_x * sum(sigma_x)
    sx_list = [nk.operator.spin.sigmax(hilbert, i) for i in range(N)]
    term_field = -h_x * sum(sx_list)
    
    # 3. Término de interacción: + J * sum(sigma_z_i * sigma_z_j / dist^alpha)
    def interaction_term(i, j):
        # Distancia con condiciones de contorno periódicas
        d = min(abs(i - j), N - abs(i - j))
      
        op_zz = nk.operator.spin.sigmaz(hilbert, i) @ nk.operator.spin.sigmaz(hilbert, j)
        return (J / d**alpha) * op_zz

    # Sumamos todos los pares (i < j)
    term_interaction = sum([interaction_term(i, j) for i in range(N) for j in range(i + 1, N)])
    
    return term_field + term_interaction