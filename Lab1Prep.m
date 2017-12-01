%%
clear all
A = [1 -1.5 0.7];
C = [1 zeros(1,11) -1];
A12 = [1 zeros(1,11) -1];
A_star = conv(A,A12);
e = randn(600,1);
y = filter(C,A_star,e);
y = y(100:end);
figure(1)
plot(y)
N = length(y);
%y = ones(1,N);
%plot y first
S = 5;
yhat = zeros(1,N-S);
for i=1:N-S
    yhat(i) = y(i+S) - y(i);
end
%INFO: A z1 & z3
%INFO: C z1 & z12
%(plot yhat + its acf&pacf and pick the values given in there?)
figure(2)
plot(yhat)
acf = acf(yhat,20,[],[],12,[]);
pacf = pacf(yhat,20);
figure(3)
stem(acf)
figure(4)
stem(pacf)
%%
%6. Statistical properties: residual has to seem to be white
%   Things to test: Acf/Pacf (sigf diff fr 0), testCumPer, signChange,
%   (resid norm distr.)

%7. Biased auto-covar fct. (p.45)
%   Properties: biased (asymp. unbiased), 0 <= r(0) >= r(tau), even fct.
%   Rule-of-thumb: tau < N/4
%   Var: 2 * sigma_e^2 * H^(-1) = 2 * 1/Ñ * S_theta * H^(-1)

%3. 

%QUESTIONS
%   IS THIS CORRECT (VAR)?

%5. FPE, will this be brought up in lectures?
