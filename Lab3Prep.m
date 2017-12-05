%% Lab3 prep exercise 1
P = [7/8 1/8; 1/8 7/8];
%what should the states be??





%% Lab3 prep exercise 2
load tar2.dat
y = tar2;
N = length(y);
a1 = 0;
a2 = 0;
A = [-a1 -a2; 1 0];
sigma2_e = 2;
sigma2_w = 1.25;
Re = [sigma2_e 0; 0 0];
Rw = [sigma2_w 0; 0 sigma2_w];
%C = 0;
Rxx_1 = var(tar2(1:10)) * eye(2);
xtt_1 = [mean(tar2(1:10)) 0]';
xsave = zeros(2,N);
for k=3:N
    C = [1 u_t]; %added input signal, insert it before running
    
    %Update
    Ryy = Rxx_1 + Rw; %dunno
    Kt = Rxx_1*C'\Ryy; %kalman?
    xtt = xtt_1; %2x1
    Rxx = Rxx_1; %2x2?
    
    
    %Save
    xsave(:,k) = xtt; %2x1
    
    %Predict
    Rxx_1 = A*Rxx_1*A' + Re; %2x2
    xtt_1 = xtt_1 + Kt*(y(k) - xtt_1); %2x1
end
plot(xsave')



