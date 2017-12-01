[y, Fs] = audioread('fa.wav');
figure(1)
plot(y)
y_samp = y(2800:3000);
figure(2)
plot(y_samp)
%%
L = 45;
cov = acf(y_samp, L);
figure(3)
plot(cov)
%%
zero = zeros(824,1);
y_samp2 = [y_samp;zero];
P = length(y_samp2);
ff_wrong = (0:P-1)/P;
ff = (0:P-1)/P - 0.5;
X_wrong = abs(fft(y_samp2,P).^2);
X = fftshift(abs(fft(y_samp2,P).^2));
figure(1)
plot(ff_wrong,X_wrong)
figure(2)
plot(ff,X)
