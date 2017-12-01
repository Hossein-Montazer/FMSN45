n = 500;
A1 = [1 -.65];
A2 = [1 .90 .78];
C = 1;
B = [0 0 0 0 .4];
e = sqrt(1.5)*randn(n + 100, 1);
w = sqrt(2)*randn(n + 200, 1);
A3 = [1 .5];
C3 = [1 -.3 .2];
u = filter(C3,A3,w);
u = u(101:end);
y = filter(C,A1,e ) + filter(B,A2, u);
u = u (101:end);
y = y (101:end);
%clear A1 A2 C B e w A3 C3
%%
figure(1)
acf(u,30,0.05,1);
figure(2)
pacf(u,30,0.05,1);
figure(3)
normplot(u)
%% Task 1
u_data = iddata(u);
u_poly = idpoly([1 0],[],[1 0 0]); %model as ARMA(1,1), actually ARMA(1,2). Can't get rid of peak at 12
%u_poly.Structure.a.Free = [0 1 zeros(1,1) 1];
u_model = pem(u_data,u_poly);
u_a = u_model.a;
u_c = u_model.c;
u_pw = filter(u_a,u_c,u);

acfpacfnorm(u_pw,30,0.05);
y_pw = filter(u_a,u_c,y);

%% Decide s, d, r
M = 40;
%cc_uy = crosscorr(u_pw,y_pw,M);
crosscorrel(u_pw,y_pw,M);
%preliminarily: s = 0, d = 4, r = 2
%s = 7, d = 4, r = 0
%%
%matrix1 = [cc_uy(3) cc_uy(2) cc_uy(1); cc_uy(4) cc_uy(3) cc_uy(2); cc_uy(5) cc_uy(4) cc_uy(3);] \ [cc_uy(4); cc_uy(5); cc_uy(6)];
%matrix1 = [cc_uy(45) cc_uy(44); cc_uy(46) cc_uy(45)] \ [cc_uy(46); cc_uy(47)];
%matrix1 = [0 0 0; cc_uy(4) 0 0; cc_uy(5) cc_uy(4) 0;] \ [cc_uy(4); cc_uy(5); cc_uy(6)];
%b0 = cc_uy(45) - matrix1(1)*cc_uy(44) - matrix1(2)*cc_uy(43);
A2 = [1 0 0]; %r=2
B = [0]; %s=1
B = [0 0 0 0 B]; %d=4
Mi = idpoly([1],[B],[],[],[A2]);
Mi.Structure.b.Free = [zeros(1,3) 1 1];
z_pw = iddata(y_pw,u_pw);
Mba2 = pem(z_pw,Mi);
present(Mba2)
v_hat = resid(Mba2,z_pw);
acfpacfnorm(v_hat.y,30,0.05);
M = 30;
crosscorrel(u_pw,v_hat.y,M);

%%
x = y - filter(Mba2.b, Mba2.f, u);
%Use standard ARMA_identification for A1 and C1
acfpacfnorm(x,30,0.05);
crosscorrel(u,x,M);
%%
x_data = iddata(x);
x_poly = idpoly([1 0],[],[]);
%x_poly.Structure.a.Free = [0 0 1 1]; %Free AR_1,3
%x_poly.Structure.c.Free = [0 1 0 1 0 0 0 0 1]; %Free MA_3
x_model = pem(x_data,x_poly);
x_a = x_model.a;
x_c = x_model.c;
e_x = filter(x_a,x_c,x);
acfpacfnorm(e_x,30,0.05);


%stuff

%%
A1 = [1 0];
A2 = [1 0 0];
B = [0 0 0 0 0]; %include b0 or not?
C = [1];
Mi = idpoly(1,B,C,A1,A2);
Mi.Structure.b.Free = [0 0 0 0 1];
z = iddata(y,u);
MboxJ = pem(z,Mi);
present(MboxJ)
ehat = resid(MboxJ,z);
acfpacfnorm(ehat.y,30,0.05);
crosscorrel(u,ehat.y,30);

%%
clear
clc
%% Task2
load tork.dat
tork = tork - repmat(mean(tork),length(tork),1);
u = tork(:,2);
y = tork(:,1);
f_s = 1/0.08;
z = iddata(y,u);
plot(z(1:300))

%%
u = u(1:300);
y = y(1:300);
acfpacfnorm(u,30,0.05);

%%
u_data = iddata(u);
u_poly = idpoly([1 0],[],[]);
%u_poly.Structure.a.Free = [0 1 zeros(1,10) 1];
u_model = pem(u_data,u_poly);
u_a = u_model.a;
u_c = u_model.c;
u_pw = filter(u_a,u_c,u);

acfpacfnorm(u_pw,30,0.05);

y_pw = filter(u_a,u_c,y);

%%
M = 40;
crosscorrel(u_pw,y_pw,M); %d = 3, r = 2, s = 2
%%
A2 = [1 0 0];
B = [0 0];
B = [0 0 0 B];
Mi = idpoly([1],[B],[],[],[A2]);
Mi.Structure.b.Free = [0 0 0 1 1];
z_pw = iddata(y_pw,u_pw);
Mba2 = pem(z_pw,Mi);
present(Mba2)
v_hat = resid(Mba2,z_pw);

acfpacfnorm(v_hat.y,30,0.05);
M = 40;
crosscorrel(u_pw,v_hat.y,M);

%%
x = y - filter(Mba2.b,Mba2.f,u);

acfpacfnorm(x,30,0.05);
crosscorrel(u,x,100);

x_data = iddata(x);
x_poly = idpoly([1 0],[],[]);
%x_poly.Structure.a.Free = [0 1 zeros(1,10) 1];
x_model = pem(x_data,x_poly);
x_a = x_model.a;
x_c = x_model.c;
e_x = filter(x_a,x_c,x);

%acfpacfnorm(e_x,30,0.05);

%%

A1 = [1 0];
A2 = [1 0 0];
B = [0 0 0 0 0]; %include b0 or not?
C = [1];
Mi = idpoly([1],B,C,A1,A2);
Mi.Structure.b.Free = [0 0 0 1 1];
z = iddata(y,u);
MboxJ = pem(z,Mi);
present(MboxJ)
ehat = resid(MboxJ,z);
acfpacfnorm(ehat.y,30,0.05);
crosscorrel(u,ehat.y,30);



%%
clear
clc
%% Task 3
load svedala
y = svedala;
%y = y-mean(y);
A = [1 -1.79 0.84];
C = [1 -0.18 -0.11];
k = 3;
[CS,AS] = equalLength(C,A);
[Fk,Gk] = deconv(conv([1 zeros(1,k-1)],CS),AS);
yhat_k = filter(Gk,C,y); %throw away samples?
y = y(max(length(Gk),length(C)):length(y)); %remove samples
yhat_k = yhat_k(max(length(Gk),length(C)):length(yhat_k)); %remove samples
%% pole testing
figure(1)
Fk_id = idpoly(Fk);
pzmap(Fk_id)
Fk_roots = roots(Fk);
Fk_mirr = 1./Fk_roots;
Fk_mirr2 = poly(Fk_mirr);
figure(2)
Fk_id2 = idpoly(Fk_mirr2);
pzmap(Fk_id2)
%% 
figure(3)
plot(yhat_k)
%yhat_var = var(yhat_k);
FkEst_err = y-yhat_k;
figure(4)
plot(FkEst_err)
figure(5)
acfpacfnorm(FkEst_err,20,0.05)
est_err = filter(1,Fk,FkEst_err); %throw away samples? make sure to choose right Fk/FkMirr
est_err = est_err(max(length(Gk),length(C)):length(est_err)); %remove samples
%plot(est_err)
err_mean = mean(FkEst_err);
%err_var = var(FkEst_err);
noise = 0.3751;
Ftot = 0;
for i = 1:length(Fk)
    Fele = Fk(i)^2;
    Ftot = Ftot + Fele;
end
err_var = noise*(Ftot);

%% solo dev
% 95% is +-2/sqrt(length(y)) if norm distr => 
%ci_95 = 2/sqrt(length(y)); %conf int 95%
ci_95 = sqrt(err_var)*1.96;
FkEst_err_acf = acf(FkEst_err,30);
nbrError = sum(abs(FkEst_err_acf)>ci_95)-1;
p_Error = nbrError/(length(FkEst_err_acf)-1);
figure(1)
plot(y)
hold on
plot(yhat_k) %they seem shifted? maybe not
hold off
figure(2)
acfpacfnorm(yhat_k,30,0.05)
figure(3)
acfpacfnorm(y,30,0.05)
figure(4)
acfpacfnorm(FkEst_err,30,0.05)

%%
clear
clc
%% Task 4
load sturup
u = sturup;
A_u = [1 -1.49 0.57];
B_u = [0 0 0 0.28 -0.26]; %Delay d = 3
C_u = [1];
%run svedala stuff
BF = conv(B_u,Fk);
[Fku,Gku] = diophantine(BF,C_u,k); %C or C_u?
uhat_k = filter(Gku,C_u,u); %throw away samples?
uhat_k = uhat_k(max(length(Gku),length(C_u)):length(uhat_k)); %remove samples
y1hat_k = filter(Gk,C_u,y); %C or Cu? throw away samples?
y1hat_k = y1hat_k(max(length(Gk),length(C_u)):length(y1hat_k)); %remove samples
yhat = y1hat_k+uhat_k;
%compare y w/ yhat
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
%% Task5
load svedala
%plot(svedala)
%periodicity 23,24,25
%S = [1 zeros(1,22) -1 -1 -1];
S = 24;
AS = [1 zeros(1,S-1) -1];
%other parameters
%A = [];
%Astar = conv(A,S);
%model_init = idpoly([],[],[],[],[]);
%model_init.Structure.a.Free = [1 zeros(1,22) 1 1 1]; %change to compensate for A
y5 = filter(AS,1,svedala); %remove seasonality
y5 = y5(S+1:length(y5)); %remove corrupted samples
figure(1)
acfpacfnorm(y5,30,0.05);
y5_poly = idpoly([1 0 0],[],[1 zeros(1,23) 0],[],[]);
y5_poly.Structure.c.Free = [0 zeros(1,23) 1];
y5data = iddata(y5);
model_init = pem(y5data,y5_poly);
y5_2 = resid(model_init,y5data);
figure(2)
acfpacfnorm(y5_2.y,30,0.05);
%%
Astar = conv(AS,model_init.a);
model2 = model_init; %create copy
model2.a = Astar; %replace A polynomial with SARIMA
svedala_data = iddata(svedala); %iddata stuff
svedala_res = resid(model2,svedala_data); %calculate residual for SARIMA model
%figure(1)
%plot(svedala_res.y)
%var_SARIMA = var(svedala_res.y)
k = 3;
A = model2.a;
C = model2.c;
[Fks,Gks] = diophantine(C,A,k);
yhat_s = filter(Gks,C,svedala);
%remove corrupted data points
figure(1)
hold on
plot(svedala)
plot(yhat_s)
hold off
figure(2)
y_res = svedala - yhat_s;
plot(y_res)
var_s = var(y_res)

%impact of filter vs resid, which one to use








