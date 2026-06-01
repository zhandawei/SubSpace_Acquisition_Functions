import torch
from botorch.models.transforms import Normalize,Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc
from test_problems import get_problems
from subspace_acquicisiton_functions import SubspaceExpectedImprovement,SubspaceqExpectedImprovement
import warnings
warnings.filterwarnings("ignore")

# 'SumSquares','Rosenbrock','DixonPrice','Ackley','Rastrigin','Griewank','Levy','Michalewicz','Schwefel'
fun_name = 'Rosenbrock'
num_vari = 4
num_initial = 2*num_vari
max_evaluation = num_initial + 256
batch_size = 4
obj_fun = get_problems(fun_name,num_vari)
bounds = obj_fun.bounds
sampler = qmc.LatinHypercube(d=num_vari)
train_x = torch.from_numpy(sampler.random(n=num_initial))*(bounds[1,:]-bounds[0,:]) + bounds[0,:]
train_y = obj_fun(train_x).unsqueeze(-1)
fmin = train_y.max() 
xmin = train_x[train_y.argmax(),:]     
iteration = 0
evaluation = train_x.shape[0]
print(f'SSEI on {num_vari}-D {fun_name}, batch size: {batch_size}, iterations: {iteration}, evaluations: {evaluation}, fmin: {-fmin.item()}')
while evaluation < max_evaluation:
    model = SingleTaskGP(train_X=train_x,
                        train_Y=train_y,
                        train_Yvar = torch.full_like(train_y, 1e-6),
                        input_transform=Normalize(d=num_vari), 
                        outcome_transform=Standardize(m=1)
                        )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)  
    # number of subspaces
    num_subspaces = min(2**num_vari-1,batch_size)
    count = 0
    subspaces = []
    while count < num_subspaces:
        rand_perm = torch.randperm(num_vari)
        rand_index = torch.randint(1,num_vari+1,(1,1)).squeeze(-1)
        subspace = torch.sort(rand_perm[0:rand_index]).values
        is_selected = []
        for i in range(count):
            is_selected.append(torch.equal(subspace,subspaces[i]))
        if sum(is_selected) == 0:                     
            subspaces.append(subspace)
            count = count + 1
    # number of points selected in each subspace
    q_list = int(batch_size/num_subspaces)*torch.ones(num_subspaces,dtype=int)
    q_list[:batch_size%num_subspaces] = q_list[:batch_size%num_subspaces] + 1
    # select q points in n different subspaces 
    new_x = xmin.repeat(batch_size,1) 
    count_q = 0                             
    for i in range(num_subspaces): 
        real_q = q_list[i]            
        if real_q == 1:
            acq_fun = SubspaceExpectedImprovement(model=model,best_f=fmin,best_x=xmin,subspace=subspaces[i])
        else:
            acq_fun = SubspaceqExpectedImprovement(model=model,
                                                   best_f=fmin,
                                                   best_x=xmin,
                                                   subspace=subspaces[i],
                                                   sampler=SobolQMCNormalSampler(torch.Size([1024]))
                                                   )
        sub_x,acq_ESSI = optimize_acqf(acq_function=acq_fun,
                                bounds = bounds[:,subspaces[i]],
                                q = real_q,
                                num_restarts=20,
                                raw_samples=1024
                            )                    
        new_x[count_q:count_q+real_q,subspaces[i]] = sub_x
        count_q = count_q + real_q     
    new_y   = obj_fun(new_x).unsqueeze(-1)
    train_x = torch.cat([train_x,new_x])
    train_y = torch.cat([train_y,new_y])
    fmin = train_y.max()    
    xmin = train_x[train_y.argmax(),:].reshape((1,num_vari))        
    evaluation = train_x.shape[0]
    iteration = iteration + 1
    print(f'SSEI on {num_vari}-D {fun_name}, batch size: {batch_size}, iterations: {iteration}, evaluations: {evaluation}, fmin: {-fmin.item()}')


