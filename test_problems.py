import torch
from torch import Tensor
from botorch.test_functions.synthetic import (
    Ackley,
    DixonPrice,
    Griewank,
    Levy,
    Michalewicz,
    Rastrigin,
    Rosenbrock,
    SyntheticTestFunction
)

class SumSquares(SyntheticTestFunction):
    def __init__(
        self,
        dim=2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        self.dim = dim
        self.continuous_inds = list(range(dim))
        if bounds is None:
            bounds = [(-5.12, 5.12) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, dtype=dtype)
        
    def _evaluate_true(self,X: Tensor) -> Tensor:
        dim = self.dim
        f = torch.sum(torch.arange(1,dim+1)*(X**2),dim=1)
        return f

class Schwefel(SyntheticTestFunction):
    def __init__(
        self,
        dim=2,
        noise_std: float | None = None,
        negate: bool = False,
        bounds: list[tuple[float, float]] | None = None,
        dtype: torch.dtype = torch.double,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            dtype: The dtype that is used for the bounds of the function.
        """
        self.dim = dim
        self.continuous_inds = list(range(dim))
        if bounds is None:
            bounds = [(-500, 500) for _ in range(self.dim)]
        self._optimizers = [tuple(420.9687 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, dtype=dtype)

    def _evaluate_true(self,X: Tensor) -> Tensor:
        dim = self.dim
        f = 418.9829*dim - torch.sum(X*torch.sin(torch.sqrt(torch.abs(X))),dim=1)
        return f


def get_problems(name: str, dim: int):
    if name == 'Ackley':
        return Ackley(dim = dim,negate=True)
    elif name == 'DixonPrice':
        return DixonPrice(dim = dim,negate=True)
    elif name == 'Griewank':
        return Griewank(dim = dim,negate=True)
    elif name == 'Levy':
        return Levy(dim = dim,negate=True)
    elif name == 'Michalewicz':
        return Michalewicz(dim = dim,negate=True)
    elif name == 'Rastrigin':
        return Rastrigin(dim = dim,negate=True)
    elif name == 'Rosenbrock':
        return Rosenbrock(dim = dim,negate=True)
    elif name == 'SumSquares':
        return SumSquares(dim = dim,negate=True)
    elif name == 'Schwefel':
        return Schwefel(dim=dim,negate=True)

