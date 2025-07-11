clearvars;clc;close all;
% objective function
fun_name = 'Rosenbrock';
% number of design variables
num_vari = 10;
% lower and upper bounds
lower_bound = -2.048*ones(1,num_vari);
upper_bound = 2.048*ones(1,num_vari);
% number of initial designs
num_initial = 10*num_vari;
% number of additional function evaluations
addition_evaluation = 12.8*num_vari;
% batch size
batch_size = 4;
max_iteration = round(addition_evaluation/batch_size);
sample_x = lhsdesign(num_initial,num_vari).*(upper_bound-lower_bound)+lower_bound;
sample_y = zeros(size(sample_x,1),1);
for ii = 1:size(sample_x,1)
    sample_y(ii) = feval(fun_name,sample_x(ii,:));
end
evaluation =  size(sample_x,1);
iteration = 1;
[fmin,ind] = min(sample_y);
xmin = sample_x(ind,:);
fprintf('ESSI on %d-D %s, batch size: %d,iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,batch_size,iteration-1,evaluation,fmin);
while iteration <= max_iteration
    % train the GP model
    GP_model = GP_Train(sample_x,sample_y,lower_bound,upper_bound,1,0.001,1000);
    % select the subspaces randomly
    % optimize the ESSI functions
    % you can use parfor here to optimize these ESSI functions in parallel
    infill_x = cell(1,batch_size);
    for ii = 1:batch_size
        subspace = randperm(num_vari,randi(num_vari));
        lower = lower_bound(subspace);
        upper = upper_bound(subspace);
        [optimized_x,max_EI] = Optimizer_GA(@(x)-Infill_ESSI(x,GP_model,fmin,xmin,subspace),length(subspace),lower,upper,10*length(subspace),100);
        infill_x{ii} = xmin;
        infill_x{ii}(subspace) = optimized_x;
    end
    infill_x = reshape(cell2mat(infill_x),num_vari,batch_size)';
    % evaluate the query points in parallel
    infill_y = zeros(size(infill_x,1),1);
    for ii = 1:size(infill_x,1)
        infill_y(ii) = feval(fun_name,infill_x(ii,:));
    end
    sample_x = [sample_x;infill_x];
    sample_y = [sample_y;infill_y];
    iteration = iteration + 1;
    evaluation = evaluation + size(infill_x,1);
    [fmin,ind] = min(sample_y);
    xmin = sample_x(ind,:);
    fprintf('ESSI on %d-D %s, batch size: %d,iteration: %d, evaluation: %d, best: %0.4g\n',num_vari,fun_name,batch_size,iteration-1,evaluation,fmin)
end


