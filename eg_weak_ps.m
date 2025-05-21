addpath(genpath('chebfun-master'))
rng(1)

%% problem specification

M = 20; % number of registry points
N = 100; % number of images
Kvec = (M)*ones(N,1); % number of keypoints per image
corrupt_prob = 0.1; % probability of corrupted permutation

K = max(Kvec); % max block size (don't change)
L = sum(Kvec); % total matrix size (don't change)

beta = 10*log(N)/N; % regularization parameter (larger means less reg, scales with log(N)/N ?)
maxIter = 200; % number of opt iters
Ncheb = 50; % initial guess for number of chebyshev polynomials

S = ceil(32*log(L)); % number of samples per iter (should NOT scale with K)

% recovery
fast_recov_flag = false; % choose fast or slow recovery
Ktilde = 2*K; % defines number of bits for robust encoding in fast recovery (at least K)

Srecov = 200; % number of samples in masked recovery

%% build cost matrix

% ground truth
count = 1;
subsets = cell(N,1);
blockInds = cell(N,1);
II = zeros(L,1);
JJ = zeros(L,1);
VV = ones(L,1);
for i=1:N
    subsets{i} = randsample(M,Kvec(i),false)';
    blockInds{i} = count:count+Kvec(i)-1;
    II(count:count+Kvec(i)-1) = blockInds{i};
    JJ(count:count+Kvec(i)-1) = subsets{i};
    count = count+Kvec(i);
end
P = sparse(II,JJ,VV,L,M);
Qtrue = P*P';

% corruption
Q = Qtrue;
for i=1:N
    for j=i+1:N
        if rand<corrupt_prob
            Pi = sparse(1:Kvec(i),randsample(M,Kvec(i),false),1,Kvec(i),M);
            Pj = sparse(1:Kvec(j),randsample(M,Kvec(j),false),1,Kvec(j),M);
            Qij = Pi*Pj';
            Q(blockInds{i},blockInds{j}) = Qij;
            Q(blockInds{j},blockInds{i}) = Qij';
        end
    end
end

C = -Q;


%% build sparse matrix for sum constraint

II = zeros(L,1);
JJ = zeros(L,1);
VV = zeros(L,1);
count = 0;
for i=1:N
    K0 = Kvec(i);
    II(count+1:count+K0) = blockInds{i};
    JJ(count+1:count+K0) = i;
    VV(count+1:count+K0) = 1/sqrt(K0);
    count = count+K0;
end
W = sparse(II,JJ,VV);


%% main optimization loop

lambda = zeros(L,1);
mu = zeros(N,1);
feasHist = zeros(maxIter,1);

tic
for iter=1:maxIter

    A0 = C - spdiags(lambda,0,L,L);
    A0 = (A0+A0')/2;
    Afun = @(x) A0*x - W*(mu.*(W'*x));

    % estimate upper and lower bounds of spectrum
    a = real(eigs(Afun,size(A0,1),1,'sa','tolerance',.1*N) - .2*N);
    b = real(eigs(Afun,size(A0,1),1,'la','tolerance',.1*N)) + .2*N;
    
    % get chebyshev approx of desired tolerance
    Ncheb = Ncheb;
    fun = @(x) exp(-(beta/2)*x);
    fun_exact = chebfun(fun,[a,b]);
    fun_cheb = chebfun(fun_exact,Ncheb);
    c = chebcoeffs(fun_cheb);
    
    z = randn(L,S);
    v = mfunv_cheb(Afun,c,z,a,b);

    % diagonal update
    d = sum(v.*v,2)/S;
    tr = sum(d);

    gr1 = d/tr - ones(L,1)/L;
    lambda = lambda - 0.5*(1/beta)*norm(gr1,1)*sign(gr1);
    lambda = lambda - mean(lambda);
    
    % block sum update
    % LOOP CAN BE PARALLELIZED (just substitute parfor)
    gr2 = zeros(N,1);
    for n=1:N
        K0 = Kvec(n);
        Vn = v(blockInds{n},:);
        w = sum(Vn',2)/sqrt(K0);
        b = sum(w.^2)/S;
        gr2(n) = 1*(b/tr - (1/L));
    end
    mu = mu - 0.5*(1/beta)*norm(gr2,1)*sign(gr2);

    feas1 = norm(gr1,1);
    feas2 = norm(gr2,1);
    feas = sqrt(feas1^2 + feas2^2);
    feasHist(iter) = feas;
    
    disp(['Iter #',num2str(iter)])
    disp(['Feasibility: ',num2str(feas)])
    disp(" ")

end
opttime = toc

figure(1);plot(feasHist);hold on



%% recover permutations

% construct final A matrix
A0 = C - spdiags(lambda,0,L,L);
A0 = (A0+A0')/2;
Afun = @(x) A0*x - W*(mu.*(W'*x));

% get chebyshev approx (now for exp(-beta*x))
a = eigs(Afun,size(A0,1),1,'sa','tolerance',.1*N) - .2*N;
b = eigs(Afun,size(A0,1),1,'la','tolerance',.1*N) + .2*N;
fun = @(x) exp(-beta*x);
fun_exact = chebfun(fun,[a,b]);
fun_cheb = chebfun(fun_exact,Ncheb);
c = chebcoeffs(fun_cheb);

unreg = true(L,1);
unreg_subsets = cell(N,1);
R = cell(N,1);
for i=1:N
    unreg_subsets{i} = 1:Kvec(i);
    R{i} = zeros(Kvec(i),1);
end

m = 0;
available_j = (1:N);

while sum(unreg) > 0

    % random j selection
    ind = randsample(length(available_j),1);

    % % j selection by minimization
    % vals = zeros(length(available_j),1);
    % for ind = 1:length(available_j)
    %     vals(ind) = full(sum(sum(Q(unreg,unreg_subsets{available_j(ind)}))));
    % end
    % [~,ind] = min(vals);

    j = available_j(ind);
    available_j(ind) = [];

    
    if fast_recov_flag
        sig = randperm(Ktilde);
        scram = sig(1:Kvec(j));
        B = 2*(de2bi(scram-1)-.5);
        d = size(B,2);
    else
        B = eye(Kvec(j));
        d = Kvec(j);
    end
    E = zeros(L,d);
    E(blockInds{j},:) = B;
    Y = mfunv_cheb(Afun,c,E,a,b);
    Y = Y/max(max(Y));

    Sj = unreg_subsets{j};
    Rj = R{j};
    for p = 1:length(Sj)
        l = Sj(p);
        m=m+1;
        Rj(l) = m;
    end
    R{j} = Rj;
    
    unreg = false(L,1);
    for i=1:N
        if i~=j
            Ri = R{i};
            bj = [B(Sj,:);zeros(1,d)];
            T = Sj;
            Si = unreg_subsets{i};
            del_inds = [];
            Yi = Y(blockInds{i},:);
            for p=1:length(Si)
                k = Si(p);
                [~,q] = min( sum((Yi(k,:)-bj).^2,2) );
                if q < size(bj,1)
                    l = T(q);
                    del_inds = [del_inds,p];
                    T(q) = [];
                    bj(q,:) = [];
                    Ri(k) = Rj(l);
                end
            end
            Si(del_inds) = [];
            R{i} = Ri;
            unreg_subsets{i} = Si;
            inds = blockInds{i};
            unreg(inds(Si)) = true;
        end
    end
    unreg_subsets{j} = [];
    unreg(blockInds{j}) = false;
end

% recover guesses Mstar, Pstar, Qstar from the registry
Mstar = m;
II = zeros(L,1);
JJ = zeros(L,1);
VV = ones(L,1);count = 1;
for i=1:N
    II(count:count+Kvec(i)-1) = blockInds{i};
    JJ(count:count+Kvec(i)-1) = R{i};
    count = count+Kvec(i);
end
Pstar = sparse(II,JJ,VV,L,Mstar);
Qstar = Pstar*Pstar';

% display error rate
err_rate = full(sum(sum(abs(Qstar-Qtrue))))/full(sum(sum(Qtrue)));
err_rate_naive = full(sum(sum(abs(Q-Qtrue))))/full(sum(sum(Qtrue)));
disp(['   Fraction of bad correspondences: ',num2str(err_rate)])
disp(['   Fraction w/o synchronization: ',num2str(err_rate_naive)])