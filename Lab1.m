%%
clear all
A1 = [1 -1.79 0.84];
C1 = [1 -0.18 -0.11];
A2 = [1 -1.79];
C2 = [1 -0.18 -0.11];
arma1 = idpoly(A1, [], C1);
arma2 = idpoly(A2, [], C2);
%%
figure(1)
pzmap(arma1)
figure(2)
pzmap(arma2)
%%
sigma2 = 1;
N = 500;
e = sqrt(sigma2)*randn(N,1);
y1 = filter(arma1.c, arma1.a, e);
y2 = filter(arma2.c, arma2.a, e);
%subplot(211)
%plot(y1)
%subplot(212)
%plot(y2)
%
m = 60;
r_theo = kovarians(arma1.c, arma1.a, m);
stem(0:m, r_theo*sigma2,'b')
hold on
r_est = covf(y1,m+1);
stem(0:m, r_est, 'r')
%%
%AR(2) seems best? rounding errors, c-poly less impact than penalty
%data = iddata(y1);
%ar_model = arx(y1,2);
%present(ar_model)
arma_model = armax(y1, [2 1]);
present(arma_model)
%e_hat = filter(arma1.a, arma1.c, y1);
%%
figure(1)
plot(e_hat)
figure(2)
normplot(e_hat)
acf = acf(e_hat,m);
pacf = pacf(e_hat,m);
figure(3)
stem(acf)
figure(4)
stem(pacf)

%%
n = 500;
%n = 1000;
A = [1 -1.35 0.43];
sigma2 = 4;
noise = sqrt(sigma2)*randn(n+100,1);
y = filter(1,A,noise);
y = y(101:end);
%%
subplot(211)
plot(y)
subplot(212)
plot(noise)
%%
n_est = floor(2/3*n);
y_est = iddata(y(1:n_est));
y_val = iddata(y(n_est+1:end));
NN = [1:10]';
V = arxstruc(y_est, y_val, NN);
n_order = selstruc(V,0);
n_aic = selstruc(V,'aic');
%%
for i=1:100
    noise = sqrt(sigma2)*randn(n+100,1);
    y = filter(1,A,noise);
    y = y(101:end);
    y_est = iddata(y(1:n_est));
    y_val = iddata(y(n_est+1:end));
    V = arxstruc(y_est, y_val, NN);
    n_order(i) = selstruc(V,0);
    n_aic(i) = selstruc(V,'aic');
end
figure(1)
hist(n_order)
figure(2)
hist(n_aic)
ar_model = arx(y,n_order(end));
ar_model.NoiseVariance
ar_model.CovarianceMatrix
present(ar_model)
%%
clear all
load data.dat
load noise.dat
data = iddata(data);
%ar1_model = arx(data,6);
%ar2_model = arx(data,2);
%rar1 = resid(ar1_model,data);
%rar2 = resid(ar2_model,data);
arma1_model = armax(data,[1,1]);
arma2_model = armax(data,[2,2]);
rar1 = resid(arma1_model,data);
rar2 = resid(arma2_model,data);
present(arma1_model)
present(arma2_model)
%%
figure(1)
subplot(311)
plot(rar1.y)
subplot(312)
plot(rar2.y)
subplot(313)
plot(noise)
e1 = rar1.y-noise;
e2 = rar2.y-noise;
figure(2)
subplot(211)
plot(e1)
subplot(212)
plot(e2)
%%

skip = 3-6;

%%
clear all
A = [1 -1.5 0.7];
C = [1 zeros(1,11) -1];
A12 = [1 zeros(1,11) -1];
A_star = conv(A,A12);
e = randn(600,1);
y = filter(C,A_star,e);
y = y(100:end);
m = 50;
figure(1)
plot(y)
figure(2)
normplot(y)
acf = acf(y,m);
pacf = pacf(y,m);
figure(3)
stem(acf)
figure(4)
stem(pacf)
%%
%create data
clear all
A = [1 -1.5 0.7];
C = [1 zeros(1,11) -0.5];
A12 = [1 zeros(1,11) -1];
A_star = conv(A,A12);
e = randn(600,1);
y = filter(C,A_star,e);
y = y(100:end);
m = 20;
y_s = filter(A12,1,y);
data = iddata(y_s);
%%
figure(1)
plot(y)
figure(2)
plotNTdist(y)
acf = acf(y,m);
pacf = pacf(y,m);
figure(3)
stem(acf)
figure(4)
stem(pacf)
%%
figure(1)
plot(y_s)
figure(2)
plotNTdist(y_s)
figure(3)
acf_s = acf(y_s,m)
figure(4)
pacf_s = pacf(y_s,m)

%stem(acf_s)

%stem(pacf_s)
%%
model_init = idpoly([1 0 0],[],[]);
model_armax = pem(data, model_init)
A_armax = model_armax.a;
%C_armax = model_armax.c;
C_armax = 1;
y_res = filter(A_armax,C_armax,y_s);
whitenessTest(y_res)
figure(1)
plot(y_res)
figure(2)
plotNTdist(y_res)
acf_res = acf(y_res,m);
pacf_res = pacf(y_res,m);
figure(3)
stem(acf_res)
figure(4)
stem(pacf_res)
%%
model_init = idpoly([1 0 0], [], [1 zeros(1,12)]);
model_init.Structure.c.Free = [zeros(1,12) 1];
model_armax = pem(data, model_init)
N = length(data.y);
signif = 2/sqrt(N);
level1 = zeros(1,m);
level2 = zeros(1,m);
for i=1:m
    level1(i) = signif;
    level2(i) = -signif;
end
A_armax = model_armax.a;
C_armax = model_armax.c;
y_res = filter(A_armax,C_armax,y_s);
whitenessTest(y_res)
figure(1)
plot(y_res)
figure(2)
plotNTdist(y_res)
acf_res = acf(y_res,m);
pacf_res = pacf(y_res,m);
figure(3)
stem(acf_res)
hold on
plot(level1,'-')
plot(level2,'-')
hold off
figure(4)
stem(pacf_res)
hold on
plot(level1,'-')
plot(level2,'-')
hold off
%%
clear all
load svedala
plot(svedala)
y = svedala;
A24 = [1 zeros(1,23) -1]; %why -1?
y_s = filter(A24,1,y);
%y_s = y;
figure(1)
plot(y_s)
figure(2)
plotNTdist(y_s)
m = 30;
acf_s = acf(y_s,m);
pacf_s = pacf(y_s,m);
figure(3)
stem(acf_s)
figure(4)
stem(pacf_s)
data = iddata(y_s);
%%
N = length(y_s);
signif = 2/sqrt(N);
level1 = zeros(1,m);
level2 = zeros(1,m);
for i=1:m
    level1(i) = signif;
    level2(i) = -signif;
end
%%
%a1, c1, a2, c2, a24, c24
model_init = idpoly([1 zeros(1,24)],[],[1 zeros(1,24)]);
%model_init = idpoly([1 0 0],[],[]);
model_init.Structure.a.Free = [0 1 1 zeros(1,21) 0];
model_init.Structure.c.Free = [0 0 0 zeros(1,21) 1];
model_armax = pem(data, model_init)
armax_a = model_armax.a;
armax_c = model_armax.c;
y_res = filter(armax_a,armax_c,y_s);
figure(1)
plot(y_res)
figure(2)
plotNTdist(y_res)
m = 30;
acf_res = acf(y_res,m);
pacf_res = pacf(y_res,m);
figure(3)
stem(acf_res)
hold on
plot(level1,'-')
plot(level2,'-')
hold off
figure(4)
stem(pacf_res)
hold on
plot(level1,'-')
plot(level2,'-')
hold off
figure(5)
whitenessTest(y_res)


