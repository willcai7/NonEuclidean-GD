rng(1)

% marginal sizes
m = 300;
n = 200;

% inverse temp
beta = 10;

eta = 1/(2*beta); % step size
maxIter = ceil(100*(beta)); % number of iterations


% sample random points with uniform marginals
x = sort(randn(m,1));
y = sort(randn(n,1));
mu = ones(m,1)/m;
nu = ones(n,1)/n;

% form random cost
C = randn(m,n);

% initialize dual variables
phi = zeros(m,1);
psi = zeros(n,1);

% prepare to launch
dualHist = zeros(maxIter,1);
feasHist = zeros(maxIter,1);

% main loop
for iter=1:maxIter
    % t = (1+sqrt(1+4*tprev^2))/2;

    A = C - phi - psi';
    X = exp(-beta*A);
    val = sum(sum(X));
    dualHist(iter) = dot(phi,mu)+dot(psi,nu)-(1/beta)*log(val);
    P = X/val;
    gr1 = mu - sum(P,2);
    gr2 = nu - sum(P,1)';
    feasHist(iter) = norm(gr1,1)+norm(gr2,1);

    phi = phi + eta*norm(gr1,1)*sign(gr1);
    psi = psi + eta*norm(gr2,1)*sign(gr2);

    phi = phi-mean(phi);
    psi = psi-mean(psi);

end

figure(1);plot(dualHist);title('Dual objective')
figure(2);semilogy(feasHist);title('Primal feasibility')
figure(3);imagesc(P);title('Pair marginal')