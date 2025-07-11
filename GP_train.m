function model = GP_Train(sample_x,sample_y,lower_bound,upper_bound,theta0,theta_lower,theta_upper)
n = size(sample_x,1);
normalized_sample_x = (sample_x - lower_bound)./(upper_bound - lower_bound);
Y = sample_y;
% optimize the theta values with in [10^a,10^b]
if  nargin > 5
    theta0 = log10(theta0);
    theta_lower = log10(theta_lower);
    theta_upper = log10(theta_upper);
    options = optimoptions('fmincon','Algorithm','sqp','MaxFunctionEvaluations',20,'OptimalityTolerance',1E-20,'StepTolerance',1E-20,'Display','off');
    theta = fmincon(@(x)concentrated_lnLikelihood(x,normalized_sample_x,Y),theta0,[],[],[],[],theta_lower,theta_upper,[],options);
    theta = 10.^theta;
else
    theta = theta0;
end
% calculate the correlation matrix
one = ones(n,1);
temp1 = sum(normalized_sample_x.^2.*theta,2)*one';
temp2 = normalized_sample_x.*sqrt(theta);
R = exp(-(temp1 + temp1'-2.*(temp2*temp2'))) + eye(n).*(10+n)*eps;
% use the Cholesky factorization
[L,p] = chol(R,'lower');
if p>0
    L = nearestSPD(R);
end
% calculate mu and sigma
mu = (one'*(L'\(L\Y)))/(one'*(L'\(L\one)));
sigma2 = ((Y-mu)'*(L'\(L\(Y-mu))))/n;
lnL = -0.5*n*log(sigma2)-sum(log(abs(diag(L))));
% output the results of the model
model.theta = theta;
model.mu = mu;
model.sigma2 = sigma2;
model.L = L;
model.lnL = lnL;
model.sample_x = sample_x;
model.normalized_sample_x = normalized_sample_x;
model.sample_y = sample_y;
model.lower_bound = lower_bound;
model.upper_bound = upper_bound;
end




function  obj = concentrated_lnLikelihood(x,X,Y)
theta = 10.^x;
% the concentrated ln-likelihood function
n = size(X,1);
one = ones(n,1);
% calculate the correlation matrix
temp1 = sum(X.^2.*theta,2)*one';
temp2 = X.*sqrt(theta);
R = exp(-(temp1 + temp1'-2.*(temp2*temp2'))) + eye(n).*(10+n)*eps;
% use the  Cholesky factorization
[L,p] = chol(R,'lower');
if p>0
    obj = 1E10;
else
    mu = (one'*(L'\(L\Y)))/(one'*(L'\(L\one)));
    sigma2 = ((Y-mu)'*(L'\(L\(Y-mu))))/n;
    lnL = -0.5*n*log(sigma2)-sum(log(abs(diag(L))));
    obj = -lnL;
end
end



function L = nearestSPD(A)
% nearestSPD - the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
% usage: Ahat = nearestSPD(A)
%
% From Higham: "The nearest symmetric positive semidefinite matrix in the
% Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
% where H is the symmetric polar factor of B=(A + A')/2."
%
% http://www.sciencedirect.com/science/article/pii/0024379588902236
%
% arguments: (input)
%  A - square matrix, which will be converted to the nearest Symmetric
%    Positive Definite Matrix.
%
% Arguments: (output)
%  Ahat - The matrix chosen as the nearest SPD matrix to A.

% symmetrize A into B
B = (A + A')/2;
% Compute the symmetric polar factor of B. Call it H.
% Clearly H is itself SPD.
[~,Sigma,V] = svd(B);
H = V*Sigma*V';
% get Ahat in the above formula
Ahat = (B+H)/2;
% ensure symmetry
Ahat = (Ahat + Ahat')/2;
% test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
p = 1;
k = 0;
while p ~= 0
    [L,p] = chol(Ahat,'lower');
    k = k + 1;
    if p ~= 0
        % Ahat failed the chol test. It must have been just a hair off,
        % due to floating point trash, so it is simplest now just to
        % tweak by adding a tiny multiple of an identity matrix.
        mineig = min(eig(Ahat));
        Ahat = Ahat + (-mineig*k.^2 + eps(mineig))*eye(size(A));
    end
end
end









