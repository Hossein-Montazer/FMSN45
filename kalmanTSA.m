function xsave = kalmanTSA(A,Re,Rw,Ct,data)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
y = data;
N = length(y);
%A = [1 0;0 1];
sigma2_e = 1;
sigma2_w = 1.25;
%Re = [sigma2_e 0; 0 0];
%Rw = [sigma2_w 0; 0 sigma2_w];
%Rw = sigma2_w;
%C = [1 u];
Rxx_1 = var(data(1:5)) * eye(2);
xtt_1 = [mean(data(1:5)) 0]';
xsave = zeros(2,N);
for k=3:N
    %C = [-y(k-1) -y(k-2)]; %ex 8.3, 8.9, 8.12
    C = [1 Ct(k)];
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
end

