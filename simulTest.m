A_y = [1 0.78 0.39];
C_y = [1 0.8 0.12 0.23];
A_u = [1 0.63];
C_u = [1 0.4 0.3];
samp_mod = 1500;
samp_val = 2500;
yrandM = randn(samp_mod,1);
urandM = randn(samp_mod,1);
yrandV = randn(samp_val,1);
urandV = randn(samp_val,1);
y_mod = filter(A_y,C_y,yrandM);
u_mod = filter(A_u,C_u,urandM);
y_val = filter(A_y,C_y,yrandV);
u_val = filter(A_u,C_u,urandV);

%% Task C - Kalman with external signal
A = [eye(8)];%a1,a24,a25,c1,c2,c3,c22,c24,c25,b0
%B = [ones(5,1)];%for external input
N = length(y_mod);
nparam = length(A);%+length(B)?;
e = zeros(1,N);
eu = zeros(1,N);
sigma2_w = var(y_mod)*4;
sigma2_e = 0.0001;
Re = [sigma2_e*eye(nparam)];
Rw = sigma2_w;

Rxx_1 = 1 * eye(nparam); %how much we trust initial values
xtt_1 = [zeros(nparam,1)]; %initial values to estimate, one for each parameter
xtt_1 = [x_result];
pstep = 1;
y_conc = [y_mod(length(y_mod)-pstep+1:end); y_val];
M = length(y_conc);
e_conc = zeros(1,M);
eu_conc = zeros(1,M);
u_conc = [u_mod(length(u_mod)-pstep+1:end); u_val]; %u_valid;%what is this supposed to be?
xsave = zeros(nparam,M);
ysave=zeros(M-pstep+1,1);
for k=4:M
    y_pred = zeros(24+1+pstep,1);
    e_pred = zeros(3+1+pstep,1);
    u_pred = zeros(1+1+pstep,1);
    eu_pred = zeros(2+1+pstep,1);
    for l=1:4 %get init values for this loop
        y_pred(l) = y_conc(l-4+k);
    end
    for l=1:4
        e_pred(l) = e_conc(l-4+k);
    end
    for l=1:2
        u_pred(l) = u_conc(l-2+k);
    end
    for l=1:2
        u_pred(l) = u_conc(l-2+k);
    end
    C = [-y_conc(k-1) -y_conc(k-2) e_conc(k-1) e_conc(k-2) e_conc(k-3) u_conc(k-1) eu_conc(k-1) eu_conc(k-2)];%residual=e
    %Update
    Ryy = C*Rxx_1*C' + Rw; %dunno
    Kt = (Rxx_1*C')/Ryy; %kalman?
    xtt = xtt_1 + (Kt*(y_conc(k) - C*xtt_1)); %2x1
    Rxx = (eye(nparam)-Kt*C)*Rxx_1; %2x2?
    
    
    %Save
    xsave(:,k) = xtt; %2x1
    e_conc(k) = y_conc(k)-C*xtt_1;
    
    %Predict
    Rxx_1 = A*Rxx*A' + Re; %2x2
    xtt_1 = A*xtt; %2x1 A*xtt + B*u_t
    
    for j=1:pstep %1:k-1 step predictions for y_t+j
    C_temp = [-y_pred(j-1+4) -y_pred(j-2+4) e_pred(j-1+4) e_pred(j-2+4) e_pred(j-3+4) u_pred(j-1+3) eu_pred(j-1+3) eu_pred(j-2+3)];
    y_pred(4+j) = C_temp*xsave(:,k);
        %if j==pstep
        %    y_pred = y_pred();
        %end
    end
    C_pred = [-y_pred(4-1+pstep) -y_pred(4-2+pstep) e_pred(4-1+pstep) e_pred(4-2+pstep) e_pred(4-3+pstep) u_pred(3-1+pstep) eu_pred(3-1+pstep) eu_pred(3-2+pstep)];
    y_pred = y_pred(pstep:end);
    ysave(k) = C_pred*xsave(:,k);
end

figure(1)
plot(xsave')
figure(2)
acfpacfnorm(e_conc,50,0.05)
x_result = xsave(:,M);
figure(3)
hold on
plot(y_val)
plot(ysave)
figure(4)
whitenessTest(e_conc')