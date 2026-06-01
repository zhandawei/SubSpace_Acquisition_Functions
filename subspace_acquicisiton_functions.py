import torch
from torch import Tensor
from botorch.acquisition.analytic import AnalyticAcquisitionFunction,_scaled_improvement,_ei_helper
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.sampling.base import MCSampler
from botorch.models.model import Model
from botorch.utils.probability.utils import ndtr as Phi
from botorch.acquisition.objective import PosteriorTransform
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import average_over_ensemble_models,t_batch_mode_transform
from typing import Optional
import math



class SubspaceExpectedImprovement(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        best_x: float | Tensor,
        subspace: int | Tensor,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ):
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("best_x", torch.as_tensor(best_x))
        self.register_buffer("subspace", torch.as_tensor(subspace))
        self.maximize = maximize
    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models

    def forward(self, X: Tensor) -> Tensor:
        input_X = self.best_x.repeat(X.shape[0],1,1)
        input_X[:,:,self.subspace] = X
        mean, sigma = self._mean_and_sigma(input_X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)  
        return (sigma * _ei_helper(u)).squeeze(-1)
    
class SubspaceProbabilityOfImprovement(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        best_x: float | Tensor,
        subspace: int | Tensor,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ):
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("best_x", torch.as_tensor(best_x))
        self.register_buffer("subspace", torch.as_tensor(subspace))
        self.maximize = maximize
    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models

    def forward(self, X: Tensor) -> Tensor:
        input_X = self.best_x.repeat(X.shape[0],1,1)
        input_X[:,:,self.subspace] = X
        mean, sigma = self._mean_and_sigma(input_X)
        u = _scaled_improvement(mean, sigma, self.best_f, self.maximize)  
        return (Phi(u)).squeeze(-1)


class SubspaceUpperConfidenceBound(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: float | Tensor,
        best_x: float | Tensor,
        subspace: int | Tensor,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ):
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("best_x", torch.as_tensor(best_x))
        self.register_buffer("subspace", torch.as_tensor(subspace))
        self.maximize = maximize
    @t_batch_mode_transform(expected_q=1)
    @average_over_ensemble_models

    def forward(self, X: Tensor) -> Tensor:
        input_X = self.best_x.repeat(X.shape[0],1,1)
        input_X[:,:,self.subspace] = X
        mean, sigma = self._mean_and_sigma(input_X)
        ucb = mean + self.beta.sqrt() * sigma
        return ucb.squeeze(-1)




class SubspaceqExpectedImprovement(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        best_x: float | Tensor,
        subspace: int | Tensor,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        self.sampler = sampler
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("best_x", torch.as_tensor(best_x))
        self.register_buffer("subspace", torch.as_tensor(subspace))

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        input_X = self.best_x.repeat(X.shape[0],X.shape[1],1)
        input_X[:,:,self.subspace] = X
        posterior = self.model.posterior(input_X)
        samples = self.get_posterior_samples(posterior).squeeze(-1)  # n x b x q
        qEI = torch.mean((torch.max(samples,axis = 2).values - self.best_f).clamp_min(0),axis = 0)
        return qEI

class SubspaceqProbabilityOfImprovement(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        best_f: float | Tensor,
        best_x: float | Tensor,
        subspace: int | Tensor,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        self.sampler = sampler
        self.register_buffer("best_f", torch.as_tensor(best_f))
        self.register_buffer("best_x", torch.as_tensor(best_x))
        self.register_buffer("subspace", torch.as_tensor(subspace))

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        input_X = self.best_x.repeat(X.shape[0],X.shape[1],1)
        input_X[:,:,self.subspace] = X
        posterior = self.model.posterior(input_X)
        samples = self.get_posterior_samples(posterior).squeeze(-1)  # n x b x q
        # qPI = torch.mean((torch.max(samples,axis = 2).values > self.best_f).float(),axis = 0)
        qPI = torch.mean((torch.max(samples,axis = 2).values - self.best_f).sign().clamp_min(0),axis = 0)
        return qPI


class SubspaceqUpperConfidenceBound(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: float,
        best_x: float | Tensor,
        subspace: int | Tensor,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        self.sampler = sampler
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("best_x", torch.as_tensor(best_x))
        self.register_buffer("subspace", torch.as_tensor(subspace))

    @t_batch_mode_transform()
    @average_over_ensemble_models
    def forward(self, X: Tensor) -> Tensor:
        self.beta_prime = math.sqrt(self.beta * math.pi / 2)
        input_X = self.best_x.repeat(X.shape[0],X.shape[1],1)
        input_X[:,:,self.subspace] = X
        posterior = self.model.posterior(input_X)
        samples = self.get_posterior_samples(posterior).squeeze(-1)  # n x b x q
        mean = samples.mean(dim=0) 
        qUCB = torch.mean(torch.max(mean + self.beta_prime*(samples -mean).abs(),axis=2).values,axis=0)
        return qUCB
