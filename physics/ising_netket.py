import netket as nk
from netket.operator import LocalOperator
from netket.hilbert import AbstractHilbert

def get_Ising(
    N: int, 
    J: float, 
    h_x: float = 1.0, 
    dimensions: int = 1
) -> LocalOperator:
    """
    Construye el Hamiltoniano de Ising Básico (Vecinos más cercanos) usando la clase nativa de NetKet.
    
    Args:
        N: Número total de espines. (Para 2D, debe ser un cuadrado perfecto, ej. 16 para 4x4).
        J: Intensidad de interacción. 
           (J < 0 -> Fase Ferromagnética FM | J > 0 -> Fase Antiferromagnética AFM)
        h_x: Intensidad del campo transversal (por defecto 1.0).
        dimensions: 1 para cadena 1D, 2 para cuadrícula 2D con PBC.
    """
    
    
    if dimensions == 1:
        graph = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)
    elif dimensions == 2:
        
        L = int(round(N**0.5))
        if L * L != N:
            raise ValueError(f"Para dimensions=2, N debe ser un cuadrado perfecto (ej. 9, 16, 25). Has pasado N={N}.")
        graph = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
    else:
        raise ValueError("Solo soportamos dimensions=1 o dimensions=2.")

    # 2. Definimos el espacio de Hilbert
    hi = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)

    # 3. Construimos el Ising Nativo de NetKet (Resuelve el "Modelo de Ising Básico")
    # H = J * sum(<i,j>) Z_i Z_j - h_x * sum(i) X_i
    H_ising = nk.operator.Ising(hilbert=hi, graph=graph, h=h_x, J=J)
    
    return hi, H_ising