%% Lab3 prep exercise 1
P = [7/8 1/8; 1/8 7/8];
n = 1000;
states = [3 8];
u = Markov(P,n,states);
sum(u>5)




%% Lab3 prep exercise 2
clear all
load tar2.dat
y = tar2;
N = length(y);
A = [1 0;0 1];
sigma2_e = 0.001; %Ej modellerat brus/vitt brus från modell
sigma2_w = 10; %Brus vid observationer
Re = [sigma2_e 0; 0 0];
%Rw = [sigma2_w 0; 0 sigma2_w];
Rw = sigma2_w;
%C = 0;
Rxx_1 = var(tar2(1:5)) * eye(2);
xtt_1 = [mean(tar2(1:5)) 0]';
xsave = zeros(2,N);
for k=3:N
    C = [-y(k-1) -y(k-2)]; %ex 8.3, 8.9, 8.12
    
    %Update
    Ryy = C*Rxx_1*C' + Rw; %dunno
    Kt = (Rxx_1*C')/Ryy; %kalman?
    xtt = xtt_1 + (Kt*(y(k) - C*xtt_1)); %2x1
    Rxx = (eye(2)-Kt*C)*Rxx_1; %2x2?
    
    
    %Save
    xsave(:,k) = xtt; %2x1
    
    %Predict
    Rxx_1 = A*Rxx*A' + Re; %2x2
    xtt_1 = A*xtt; %2x1 A*xtt + B*u_t
end
plot(xsave')



