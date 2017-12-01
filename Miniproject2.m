clear all
A = [1; -1.29; 0.84; -0.4; 0.3];
C = [1; -0.81; -0.11];
N = 100;
w_t = randn(N,1);
u = filter(C,A,w_t);
lag = 20;
figure(1)
plot(u)
figure(2)
acf = acf(u,lag);
stem(acf)
figure(3)
pacf = pacf(u,lag);
stem(pacf)
%%
clear all
A = [1; -1.79; 0.74];
C = [1; -0.81; -0.11];
rootsA = roots(A);
rootsC = roots(C);
N = 1000;
w_t = randn(N,1);
u = filter(C,A,w_t);
plot(u)
%%
clear all
rootsA = [1.001 0.2+0.31i 0.2-0.31i];
rootsC = [1.0001 0.2+0.31i 0.2-0.31i];
A = poly(rootsA);
C = poly(rootsC);
N = 10000;
w_t = randn(N,1);
u = filter(C,A,w_t);
lag = 20;
figure(1)
plot(u)
%%
figure(2)
acf = acf(u,lag);
stem(acf)
figure(3)
pacf = pacf(u,lag);
stem(pacf)

%%
clear all
load week2data %gives y
figure(1)
plot(y)
mean = mean(y);
y0 = y-mean;
L = 25;
racf = acf(y0,L);
rpacf = pacf(y0,L);
r = covf(y0,L);
figure(2)
plot(0:L-1,r)
figure(3)
stem(racf)
figure(4)
stem(rpacf)