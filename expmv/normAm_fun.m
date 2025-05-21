function [c,mv] = normAm_fun(Afun,n,m)
%NORMAM   Estimate of 1-norm of power of matrix.
%   NORMAM(A,m) estimates norm(A^m,1).
%   If A has nonnegative elements the estimate is exact.
%   [C,MV] = NORMAM(A,m) returns the estimate C and the number MV of
%   matrix-vector products computed involving A or A^*.

%   Reference: A. H. Al-Mohy and N. J. Higham, A New Scaling and Squaring
%   Algorithm for the Matrix Exponential, SIAM J. Matrix Anal. Appl. 31(3):
%   970-989, 2009.

%   Awad H. Al-Mohy and Nicholas J. Higham, September 7, 2010.

t = 1; % Number of columns used by NORMEST1.


    [c,v,w,it] = normest1(@afun_power,t);
    mv = it(2)*t*m;



  function Z = afun_power(flag,X)
       %AFUN_POWER  Function to evaluate matrix products needed by NORMEST1.


       if isequal(flag,'dim')
          Z = n;
       elseif isequal(flag,'real')
          Z = true;
       else

           if isequal(flag,'notransp') || isequal(flag,'transp')

               Y = X;
               for i = 1:m
                   Y = Afun(X);
               end
               Z = Y;

           end

       end

  end
end
