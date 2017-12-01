load svedala
figure(1)
plot(svedala)
a = 5 * 17 * 2^4;
N = length(svedala);
S = 24;
yhat = zeros(N-S,1);
for i=1:N-S
    yhat(i) = svedala(i+S) - svedala(i);
end
figure(2)
plot(yhat)
acf = acf(yhat,20);
pacf = pacf(yhat,20);
figure(3)
stem(acf)
figure(4)
stem(pacf)

%HOW TO ARMA, can assume AR => MA => AR => ...
%But in the end, how to estimate all at once?
%
%How to "subtract" an AR/MA-process from data