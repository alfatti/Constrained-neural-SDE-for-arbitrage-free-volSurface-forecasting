"""
volsurface_cdg: Arbitrage-free conditional forecasting of implied volatility
surfaces via constrained neural SDEs and Doob h-transform guidance.

Pipeline:
    1. Simulate Heston option price surfaces (heston.py)
    2. Build static arbitrage constraints A c >= b (constraints.py)
    3. Decode factors and pull back the polytope to factor space (factors.py)
    4. Build exogenous context vector Y_k from history (context.py)
    5. Train a constrained neural SDE conditioned on (xi, Y) (nsde.py)
    6. Optionally train a CDG-ML guidance h-function (cdg.py)
    7. Simulate (guided or unguided), reconstruct prices, invert to IV
"""

from . import heston, constraints, factors, context, nsde, cdg

__all__ = ['heston', 'constraints', 'factors', 'context', 'nsde', 'cdg']
