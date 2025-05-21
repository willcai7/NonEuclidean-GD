addpath(genpath('lobpcg'))
addpath(genpath('expmv'))

rng(2)

n = 1000; % graph size
p = 3/n; % Erdos-Renyi edge probability
beta = 10; % reg paramater

N = ceil(25*log(n)); % number of stochastic matvecs
eta = 1/beta; % step size
maxIter = ceil(40*beta); % number of iterations
Nfinal = ceil(N*maxIter/10); % number of stochastic matvecs in postprocessing step

%generate Erdos-Renyi graph
I = [];
J = [];
for i = 2:n
    ne = binornd(i-1,p);
    inds = randsample(i-1,ne);
    I = [I;i*ones(ne,1)];
    J = [J;inds];
end
II = [I;J];
JJ = [J;I];
VV = ones(size(II));
Adj = sparse(II,JJ,VV,n,n);
C = Adj;

% initial guess
lambda = zeros(n,1);

% prepare to launch
lambdaHist = zeros(n,maxIter); % will store dual variable history
lambdaHist2 = zeros(n,maxIter); % will store history of running averages
dualHist = zeros(maxIter,1); % will store dual objective history
feasHist = zeros(maxIter,1); % will store primal feasibility history

tic
% main loop
for iter = 1:maxIter

    % effective cost matrix
    A = C - spdiags(lambda,0,n,n);
    
    % random exponential matvecs
    z = randn(n,N);
    M = select_taylor_degree(A,[],[],[],'half',[],[],true);
    v = expmv(-beta/2,A,z,[],'half',M);
    
    % trace and normalized diagonal estimate
    a = sum(v.^2,2)/N;
    tr = sum(a);
    a = a/sum(a);
    
    % gradient
    gr = ones(n,1)/n - a;

    % l infinity gradient ascent step    
    lambda = lambda + eta*norm(gr,1)*sign(gr);

    % project to zero-mean
    lambda = lambda-mean(lambda);

    % measure primal feasibility error (diagonal constraint error)
    feasHist(iter) = norm(a-(1/n),1);

    % measure dual objective
    dualHist(iter) = -(1/beta)*log(tr);
    
    % store dual variable history
    lambdaHist(:,iter) = lambda;
    lambdaHist2(:,iter) = mean(lambdaHist(:,round(iter/2):iter),2);

end
optTime = toc;

disp(['Optim time: ',num2str(optTime),' seconds'])

figure(1);plot(dualHist);title('Dual objective');hold on
figure(2);;plot(feasHist);title('Primal feasibility');hold on

% get lowest eig of effective cost matrix
A = C - spdiags(lambda,0,n,n);
[u,mu,failureFlag,lambdaHistory,residualNormsHistory] = ...
    lobpcg(randn(n,1),A,[],[],[],1e-8,1000,0);

% use to compute lower bound
lb = (sum(lambda)+mu*n);

% get upper bound as in GW
Z = randn(n,Nfinal);
Y = expmv(-beta/2,A,Z,[],'half',M);X = sign(Y);
vals = sum(X.*(C*X),1);
ub = mean(vals);

% approx ratio
disp(['Approx ratio: ',num2str(ub/lb)])