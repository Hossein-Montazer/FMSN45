%%
r = [0.3+0.2*1i 0.3-0.2*1i].';
A = poly(r);
r = roots(A);
%%
A3 = conv(A1,A2);
%%
m = 30;
A = [1; -1.79; 0.84]';
C = [1; -0.81; -0.11]';
r = kovarians(C,A,m);
figure(1)
plot(0:m,r)
figure(2)
plot(0:m,r/r(1))
figure(3)
stem(0:m,r/r(1))

%%
n = 2^8;
A = [1; -1.79; 0.84];
C = [1; -0.81; -0.11];
[H,w] = freqz(C,A,n);
R2 = abs(H).^2;
f2 = w/(2*pi);
[R,f]=spekt(C,A,n);
figure(1)
plot(f,R)
figure(2)
semilogy(f,R)

%%
sigma2 = 1;
A = [1; -1.79; 0.84]';
C = [1; -0.81; -0.11]';
Ralt=idfrd(idpoly(A,[],C,[],[],sigma2));
%Rf = Ralt.Frequency;
%Rs = Ralt.SpektrumData;
ffplot(Ralt)

%%
n = 2^10;
e = randn(n,1);
A = [1; -1.79; 0.84];
C = [1; -0.81; -0.11];
Y = filter(C,A,e);
plot(Y)

%%
%armagui
alpha = 0.98;
theta = pi*0.25;
poles = [alpha theta];
modell.A=polypolar(poles);
modell.C=[1];
%%
brus = randn(500,1);
figure(1)
plot(brus)

mean = mean(brus);
std = std(brus);
figure(2)
hist(brus,30)

rbrus=covf(brus,25);
figure(3)
stem(0:24,rbrus)

%%
load data1.dat
load data2.dat
figure(1)
subplot(2,1,1)
plot(data1)
subplot(2,1,2)
plot(data2)

mean1 = mean(data1);
data11 = data1-mean1;
mean2 = mean(data2);
data21 = data2-mean2;
rdata1 = covf(data11,25);
rdata2 = covf(data21,25);

figure(2)
subplot(2,1,1)
plot(0:24,rdata1)
subplot(2,1,2)
plot(0:24,rdata2)

