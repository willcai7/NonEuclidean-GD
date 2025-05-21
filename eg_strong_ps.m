addpath(genpath('chebfun-master'))
rng(1)

%% problem specification

M = 50; % number of registry points
N = 2*M; % number of images
Kvec = 10 + 0*randi(3,N,1); % number of keypoints per image
corrupt_prob = 0.15; % probability of corrupted permutation

K = max(Kvec); % max block size (don't change)
L = sum(Kvec); % total matrix size (don't change)

beta = 10*log(N)/N; % regularization parameter (larger means less reg, scales with log(N)/N ?)
maxIter = 100; % number of opt iters
Ncheb = 50; % chebyshev order

S = ceil(8*K*log(N)); % number of samples per iter (should scale with K)


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


%% build indexing for fast construction of block diagonal mats

II = zeros(sum(Kvec.^2),1);
JJ = zeros(sum(Kvec.^2),1);
count = 0;
for i=1:N
    K0 = Kvec(i);
    mat = repmat((1:K0)',1,K0);
    I = mat(:);
    mat = mat';
    J = mat(:);
    II(count+1:count+K0^2) = I+sum(Kvec(1:i-1));
    JJ(count+1:count+K0^2) = J+sum(Kvec(1:i-1));
    count = count+K0^2;
end


%% main optimization loop

LambdaBlocks = cell(N,1);
DeltaBlocks = cell(N,1);
for i=1:N
    LambdaBlocks{i} = zeros(Kvec(i),Kvec(i));
    DeltaBlocks{i} = zeros(Kvec(i),Kvec(i));
end
feasHist = zeros(maxIter,1);

tic
for iter=1:maxIter

    LambdaVec = zeros(sum(Kvec.^2),1);
    count = 0;
    for i=1:N
        Lambda_i = LambdaBlocks{i};
        K0 = Kvec(i);
        LambdaVec(count+1:count+K0^2) = Lambda_i(:);
        count = count+K0^2;
    end
    LambdaBlockDiag = sparse(II,JJ,LambdaVec);
    LambdaBlockDiag = LambdaBlockDiag - spdiags(trace(LambdaBlockDiag)*ones(L,1)/L,0,L,L);

    A = C - LambdaBlockDiag;
    A = (A+A')/2;
    Afun = @(x) A*x;

    % estimate upper and lower bounds of spectrum
    a = eigs(Afun,size(A,1),1,'sa','tolerance',.1*N) - .2*N;
    b = eigs(Afun,size(A,1),1,'la','tolerance',.1*N) + .2*N;
    
    % get chebyshev approx of desired tolerance
    fun = @(x) exp(-(beta/2)*x);
    fun_exact = chebfun(fun,[a,b]);
    fun_cheb = chebfun(fun_exact,Ncheb);
    c = chebcoeffs(fun_cheb);
    
    z = randn(L,S);
    v = mfunv_cheb(Afun,c,z,a,b);
    tr = sum(sum(v.*v))/S;
    
    % LOOPS CAN BE PARALLELIZED (just substitute parfor)
    
    feas = 0;
    for n=1:N
        K0 = Kvec(n);
        Vn = v(blockInds{n},:);
        Bhat = Vn*Vn'/S;
        mat = Bhat/tr - eye(K)/L;
        mat = (mat+mat')/2;
        feas = feas + sum(svd(mat));
        [uu,ee] = eig(mat);
        DeltaBlocks{n} = uu*sign(ee)*uu';
    end
    
    for n=1:N
        LambdaBlocks{n} = LambdaBlocks{n} - (1/beta)*feas*DeltaBlocks{n};
    end

    feasHist(iter) = feas;

    disp(['Iter #',num2str(iter)])
    disp(['Feasibility: ',num2str(feas)])
    disp(" ")

end
opttime = toc

figure(1);plot(feasHist(1:end));hold on

%% recover permutations

% construct final A matrix
LambdaVec = zeros(sum(Kvec.^2),1);
count = 0;
for i=1:N
    Lambda_i = LambdaBlocks{i};
    K0 = Kvec(i);
    LambdaVec(count+1:count+K0^2) = Lambda_i(:);
    count = count+K0^2;
end
LambdaBlockDiag = sparse(II,JJ,LambdaVec);
A = C - LambdaBlockDiag;
A = (A+A')/2;
Afun = @(x) A*x;

% get chebyshev approx (now for exp(-beta*x))
a = eigs(Afun,size(A,1),1,'sa','tolerance',.1*N) - .2*N;
b = eigs(Afun,size(A,1),1,'la','tolerance',.1*N) + .2*N;
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

    E = zeros(L,Kvec(j));
    E(blockInds{j},:) = eye(Kvec(j));
    Y = mfunv_cheb(Afun,c,E,a,b);
    Y = Y / max(max(Y));

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
            tmp = eye(Kvec(j));
            ej = [tmp(Sj,:);zeros(1,Kvec(j))];
            T = Sj;
            Si = unreg_subsets{i};
            del_inds = [];
            Xij = Y(blockInds{i},:);
            for p=1:length(Si)
                k = Si(p);
                [~,q] = min( sum((Xij(k,:)-ej).^2,2) );
                if q < size(ej,1)
                    l = T(q);
                    del_inds = [del_inds,p];
                    T(q) = [];
                    ej(q,:) = [];
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
disp(['Fraction of bad correspondences: ',num2str(err_rate)])
disp(['   Fraction w/o synchronization: ',num2str(err_rate_naive)])