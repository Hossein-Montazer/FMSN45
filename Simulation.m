%% Simulation of an ARMA-process
A = [1 -1.79 0.84];
C = [1 -0.18 -0.11 0.8];
ARMA_poly = idpoly(A, [], C);
pzmap(ARMA_poly)
sigma2 = 4;
N = 1000;
e = sqrt(sigma2)*randn(N,1);
y = filter (ARMA_poly.c, ARMA_poly.a, e);
plot(y)
%% Remove dependencies (mean)
y = y - mean(y);

%% Plot ACF and PACF for the simulated data
lag = 50;
conf_int = 0.05;
acfpacfnorm(y,lag,conf_int)

%% 
A = [1 0 0];
C = [1 0 0 0];
y_poly = idpoly(A,[],C,[],[]);%A,B,C,D,F
y_poly.Structure.a.Free = [0 1 1];
y_poly.Structure.c.Free = [0 1 1 1];
y_data = iddata(y);
model_y = pem(y_data,y_poly);
resid_y = resid(y_data,model_y);
present(model_y)
acfpacfnorm(resid_y.y,lag,conf_int)
%subplot(144)
%whitenessTest(resid_y.y) %3/5 great

%% Prediction
k = 5; %steps
[Fk_y,Gk_y] = diophantine(model_y.c,model_y.a,k);
yhat_gc = filter(Gk_y,model_y.c,y);
samp_remove = max(length(Gk_y),length(C));
yhat_gc = yhat_gc(samp_remove:end); %remove samples
plot(yhat_gc,'r')
hold on
plot(y(samp_remove:end),'b')

%% Check residual of prediction
resid_gc = y(samp_remove:end) - yhat_gc;
var_gc = var(resid_gc);
acfpacfnorm(resid_gc,lag,conf_int) %MA(k-1)

%% Simulation of an external input signal
A3 = [1 0.5];
C3 = [1 -0.3 0.2];
w = randn(N,1);
u = filter(C3,A3,w);
samp_remove = max(length(A3),length(C3));
u = u(samp_remove:end);
u = u - mean(u);
plot(u)

%% Prewhitening u 
A_u = [1 0];
C_u = [1 0 0];
u_data = iddata(u);
u_poly = idpoly(A_u,[],C_u);
u_poly.Structure.a.Free = [0 1];
u_poly.Structure.c.Free = [0 1 1];
model_u = pem(u_data,u_poly);
%present(model_u)
u_pw = filter(model_u.a,model_u.c,u);
samp_remove = max(length(A_u),length(C_u));
u_pw = u_pw(samp_remove:end);
acfpacfnorm(u_pw,lag,conf_int);
%subplot(144)
%whitenessTest(u_pw)

%% prewhitening y
y_data = iddata(y);
y_pw = resid(y_data,model_u);
%crosscorrel(u_pw,y_pw.y,lag); %d=0,1,6 s=0,1 r=2
A2 = [1 0];%r=2
B = [0 0];%s=1
B = [0 0 0 0 0 0 B];%d=6
Mi = idpoly(1,B,[],[],A2);
Mi.Structure.b.Free = [zeros(1,6) 1 1];
z_pw = iddata(y_pw.y(samp_remove+2:end),u_pw);
Mba2 = pem(z_pw,Mi);
%present(Mba2)
v_hat = resid(Mba2,z_pw); % Do not expect to be white.
crosscorrel(v_hat.y,u_pw,lag) % Should be white.

%% Modelling x

x = y(3:end) - filter(Mba2.b, Mba2.f, u);
x = x(10:end);
acfpacfnorm(x,lag,conf_int)
subplot(144)
crosscorrel(x,u,50) % Should be white, but is not.

%%
x_data = iddata(x);
A_x = [1 zeros(1,20)];
C_x = [1 zeros(1,22)];
x_poly = idpoly(A_x,[],C_x);
x_poly.Structure.a.Free = [0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1]; 
x_poly.Structure.c.Free = [0 zeros(1,21) 1];
model_x = pem(x_data,x_poly);
x_resid = resid(x_data,model_x);%e
%present(model_x)
acfpacfnorm(x_resid.y,lag,conf_int) % Should be white

%% Full Estimation - Moment of Truth

lag = 50;
conf_int = 0.05;
A1 = [1 zeros(1,20)];
A2 = [1 0 0];
B = [0 0];
B = [0 0 0 0 0 0 B];
C = [1 zeros(1,22)];
Mi = idpoly(1,B,C,A1,A2);
Mi.Structure.d.Free = [0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 1];
Mi.Structure.b.Free = [zeros(1,6) 0 1];
Mi.Structure.c.Free = [0 zeros(1,21) 1];
%Mi.Structure.f.Free = [0 1 1];
z = iddata(y,u);
MboxJ = pem(z,Mi);
%present(MboxJ)
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
yhat = y1hat_k + uhat_k;
yval = y1hat_val + uhat_val;

figure(1)
hold on
plot(y(length(y)-length(yhat):length(y)))%plot y time shifted to match with yhat
plot(yhat)
hold off
