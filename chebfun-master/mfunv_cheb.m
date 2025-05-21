function w = mfunv_cheb(Afun,c,v,a,b)

N = length(c);
D = size(v,1);
K = size(v,2);

ctr = (a+b)/2;
scale = (b-a)/2;

Bfun = @(x) (Afun(x) - ctr*x)/scale;


%TBfunv = zeros(D,K,N);

for n=1:N
    if n==1
        T0 = v;
        w = c(1)*T0;
    elseif n==2
        T1 = Bfun(v);
        w = w+c(2)*T1;
    else
        T = 2*Bfun(T1) - T0;
        w = w+c(n)*T;
        T0 = T1;
        T1 = T;
    end
end

%w = sum(TBfunv .* reshape(c,1,1,N),3);