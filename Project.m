%% choose data
load utempAva_9395.dat %full zeros before 6803
temp = utempAva_9395(:,3); %90*24=2160  70*24=1680 Yr1994 starts at 8761
%plot(temp) %look at all data
temp = utempAva_9395(10921:12601,3); %start of april -> mid-june, modelset
%temp = utempAva_9395(10921:17641,3); %start of april -> end-dec, whole set
plot(temp)

%% remove mean
y = temp - mean(temp);
plot(y)

%% plot acf and pacf and normplot, set lag and conf_int
lag = 50;
conf_int = 0.05;
acfpacfnorm(y,lag,conf_int)

%% remove seasonality
A = [1 zeros(1,25)];
C = [1 zeros(1,25)];
y_poly = idpoly(A,[],C,[],[]);%A,B,C,D,F
y_poly.Structure.a.Free = [0 1 0 0 zeros(1,18) 0 0 1 1];
y_poly.Structure.c.Free = [0 1 1 1 zeros(1,18) 1 0 1 0];
y_data = iddata(y);
model_y = pem(y_data,y_poly);
resid_y = resid(y_data,model_y);
present(model_y)
acfpacfnorm(resid_y.y,lag,conf_int)
subplot(144)
whitenessTest(resid_y.y) %3/5 great
jbtest(resid_y.y) %not normal, instead t-distr

%% Prediction
k = 1; %steps to predict, choose 3 and use var(resid_gc)
[Fk_y,Gk_y] = diophantine(model_y.c,model_y.a,k);
yhat_gc = filter(Gk_y,model_y.c,y);
samp_remove = max(length(Gk_y),length(C));
yhat_gc = yhat_gc(samp_remove:length(yhat_gc)); %remove samples
plot(yhat_gc,'r')
hold on
plot(y(samp_remove:end),'b')

%% Check residual of prediction
resid_gc = y(samp_remove:end) - yhat_gc;
var_gc = var(resid_gc);
acfpacfnorm(resid_gc,lag,conf_int)

%% Choose external known input; Task B

load tvxo94.mat
load ptvxo94.mat
load tstu94.mat
load ptstu94.mat
load tid94.mat
%plot(tvxo94)
%figure(2)
%plot(ptvxo94)

vxo = tvxo94(2161:3841); %model set
stu = tstu94(2161:3841); %model set
%vxo_all = tvxo94(2161:8760); %whole set, actually to 8881 (next year)
plot(vxo)
plot(stu)
%vxo = vxo.^(1/2);
%plot(vxo_all)
%% looking at u
u = vxo - mean(vxo);
%u = stu - mean(stu);
lag = 50;
conf_int = 0.05;
acfpacfnorm(u,lag,conf_int)
figure(2)
plot(u)


%% prewhitening u

A_u = [1 zeros(1,25)];
C_u = [1 zeros(1,25)];
u_data = iddata(u);
u_poly = idpoly(C_u,[],A_u);%1       5         10        15        20        25
u_poly.Structure.a.Free = [0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];%1,2
u_poly.Structure.c.Free = [0 0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0];%3,21,25 3,6,12,21,24
model_u = pem(u_data,u_poly);
present(model_u)
u_pw = resid(u_data,model_u);
acfpacfnorm(u_pw.y,lag,conf_int)
subplot(144)
whitenessTest(u_pw.y)

%% hardcoding from above
A_u = [1 zeros(1,25)];
C_u = [1 zeros(1,25)];
u_data = iddata(u);
u_poly = idpoly(C_u,[],A_u);%1       5         10        15        20        25
u_poly.Structure.a.Free = [0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];%1,2
u_poly.Structure.c.Free = [0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 0 1 0 0 1 0];%3,9,12,15,21,24
model_u = pem(u_data,u_poly);
present(model_u)
u_pw = filter(model_u.a,model_u.c,u);
u_pw = u_pw(30:end);
acfpacfnorm(u_pw,lag,conf_int)
subplot(144)
whitenessTest(u_pw)

%% prewhitening y
y_data = iddata(y);
y_pw = resid(y_data,model_u);
%crosscorrel(u_pw,y_pw.y,lag); %d=0,1,6 s=0,1 r=2
A2 = [1 0 0];%r=2
B = [0];%s=0
B = [0 0 0 0 0 0 B];%d=6
Mi = idpoly(1,B,[],[],A2);
Mi.Structure.b.Free = [zeros(1,6) 1];
z_pw = iddata(y_pw.y(30:end),u_pw);
Mba2 = pem(z_pw,Mi)
present(Mba2)
v_hat = resid(Mba2,z_pw);
crosscorrel(v_hat.y,u_pw,lag)

%% Modelling x

x = y - filter(Mba2.b, Mba2.f, u);
x = x(10:end);
acfpacfnorm(x,lag,conf_int)
subplot(144)
crosscorrel(x,u,50)
%%
x_data = iddata(x);
A_x = [1 zeros(1,25)];
C_x = [1 zeros(1,25)];
x_poly = idpoly(A_x,[],C_x);%1       5         10        15        20        25
x_poly.Structure.a.Free = [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1];%1,24,25
x_poly.Structure.c.Free = [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1];%2,24,25
model_x = pem(x_data,x_poly);
x_resid = resid(x_data,model_x);%e
present(model_x)
acfpacfnorm(x_resid.y,lag,conf_int)


%% Full Estimation - Moment of Truth
load utempAva_9395.dat %full zeros before 6803
temp = utempAva_9395(:,3); %90*24=2160  70*24=1680 Yr1994 starts at 8761
%plot(temp) %look at all data
temp = utempAva_9395(10921:12601,3); %start of april -> mid-june, modelset
%temp = utempAva_9395(10921:17641,3); %start of april -> end-dec, whole set

load tvxo94.mat
load ptvxo94.mat
load tstu94.mat
load ptstu94.mat
load tid94.mat
vxo = tvxo94(2161:3841); %model set
stu = tstu94(2161:3841); %model set
%vxo_all = tvxo94(2161:8760); %whole set, actually to 8881 (next year)
%vxo = vxo.^(1/2);
%plot(vxo_all)
u = vxo - mean(vxo);
%u = stu - mean(stu);
y = temp - mean(temp);

lag = 200;
conf_int = 0.05;
A1 = [1 zeros(1,25)];
A2 = [1 0 0];
B = [0];
B = [0 0 0 0 0 0 B];
C = [1 zeros(1,25)];
Mi = idpoly(1,B,C,A1,A2);
Mi.Structure.d.Free = [0 1 zeros(1,22) 1 1];
Mi.Structure.b.Free = [zeros(1,6) 1];
Mi.Structure.c.Free = [0 0 1 zeros(1,19) 1 0 1 0];
Mi.Structure.f.Free = [1 0 1];
z = iddata(y,u);
MboxJ = pem(z,Mi);
present(MboxJ)
e_hat = resid(MboxJ,z);
acfpacfnorm(e_hat.y,lag,conf_int)
subplot(144)
crosscorrel(u,e_hat.y,lag)


%% Task B - Prediction
A_u = MboxJ.a;
B_u = MboxJ.b;
C_u = MboxJ.c;
k = 1;
[Fk_y,Gk_y] = diophantine(model_y.c,model_y.a,k); %Need to run prev code for model_y
BF = conv(B_u,Fk_y);
[Fk_u,Gk_u] = diophantine(BF,C_u,k);

uhat_k = filter(Gk_u,C_u,u); %throw away samples?
uhat_k = uhat_k(max(length(Gk_u),length(C_u)):length(uhat_k)); %remove samples
y1hat_k = filter(Gk_y,C_u,y); %C or Cu? throw away samples?
y1hat_k = y1hat_k(max(length(Gk_y),length(C_u)):length(y1hat_k)); %remove samples
yhat = y1hat_k+uhat_k;

figure(1)
hold on
plot(y(length(y)-length(yhat):length(y)))%plot y time shifted to match with yhat
plot(yhat)
hold off
%%
%compare prediction errors
ci_95 = 2/sqrt(length(yhat)); %conf int 95%
estErr = y(length(y)-length(yhat)+1:length(y))-yhat; %y-yhat
estErr_acf = acf(estErr,30);
nbrError = sum(abs(estErr_acf)>ci_95)-1;
p_Error = nbrError/(length(estErr_acf)-1);
figure(2)
ci_95 = 2/sqrt(length(y)); %conf int 95%
acfpacfnorm(yhat,30,0.05)
figure(3)
acfpacfnorm(y,30,0.05)
figure(4)
acfpacfnorm(estErr,30,0.05)
varEstErr = var(estErr);











%% Task C - Kalman without external signal
A = [eye(8)];%a1,a24,a25,c1,c2,c3,c22,c24
B = [ones(5,1)];%for external input
N = length(y);
nparam = length(A);%+length(B)?;
e = zeros(1,N);
sigma2_w = var(y);
sigma2_e = 0.0001;
Re = [sigma2_e*eye(nparam)];
Rw = sigma2_w;

Rxx_1 = 1 * eye(nparam); %how much we trust initial values
xtt_1 = [x_result]; %initial values to estimate, one for each parameter
xsave = zeros(nparam,N);
for k=26:N
    C = [-y(k-1) -y(k-24) -y(k-25) e(k-1) e(k-2) e(k-3) e(k-22) e(k-24)];%residual=e
    %Update
    Ryy = C*Rxx_1*C' + Rw; %dunno
    Kt = (Rxx_1*C')/Ryy; %kalman?
    xtt = xtt_1 + (Kt*(y(k) - C*xtt_1)); %2x1
    Rxx = (eye(nparam)-Kt*C)*Rxx_1; %2x2?
    
    
    %Save
    xsave(:,k) = xtt; %2x1
    e(k) = y(k)-C*xtt_1;
    
    %Predict
    Rxx_1 = A*Rxx*A' + Re; %2x2
    xtt_1 = A*xtt; %2x1 A*xtt + B*u_t
end

figure(1)
plot(xsave')
figure(2)
acfpacfnorm(e,50,0.05)
x_result = xsave(:,N);

%% Task C - Kalman with external signal
A = [eye(10)];%a1,a24,a25,c1,c2,c3,c22,c24,c25,b0
%B = [ones(5,1)];%for external input
N = length(y);

nparam = length(A);%+length(B)?;
e = zeros(1,N);
sigma2_w = var(y)*2;
sigma2_e = 0.0001;
Re = [sigma2_e*eye(nparam)];
Rw = sigma2_w;

Rxx_1 = 1 * eye(nparam); %how much we trust initial values
%xtt_1 = [zeros(nparam,1)]; %initial values to estimate, one for each parameter
xtt_1 = [x_result];
xsave = zeros(nparam,N);
for k=26:N
    C = [-y(k-1) -y(k-24) -y(k-25) e(k-1) e(k-2) e(k-3) e(k-22) e(k-24) e(k-25) u(k-6)];%residual=e
    %Update
    Ryy = C*Rxx_1*C' + Rw; %dunno
    Kt = (Rxx_1*C')/Ryy; %kalman?
    xtt = xtt_1 + (Kt*(y(k) - C*xtt_1)); %2x1
    Rxx = (eye(nparam)-Kt*C)*Rxx_1; %2x2?
    
    
    %Save
    xsave(:,k) = xtt; %2x1
    e(k) = y(k)-C*xtt_1;
    
    %Predict
    Rxx_1 = A*Rxx*A' + Re; %2x2
    xtt_1 = A*xtt; %2x1 A*xtt + B*u_t
end

figure(1)
plot(xsave')
figure(2)
acfpacfnorm(e,50,0.05)
x_result = xsave(:,N);








