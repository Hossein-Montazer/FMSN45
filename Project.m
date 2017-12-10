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
k = 4; %steps to predict, choose 3 and use var(resid_gc)
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

%% Choose external known input

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
%u = vxo - mean(vxo);
u = stu - mean(stu);
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
u_poly.Structure.a.Free = [0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
u_poly.Structure.c.Free = [0 0 0 1 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0];
model_u = pem(u_data,u_poly);
present(model_u)
u_pw = filter(model_u.a,model_u.c,u);
u_pw = u_pw(30:end);
acfpacfnorm(u_pw,lag,conf_int)
subplot(144)
whitenessTest(u_pw)

%% prewhitening y

y_pw = resid(y_data,model_u);
%crosscorrel(u_pw.y,y_pw.y,lag); %d=1,s=0,r=2
A2 = [1 0];%r=1
B = [0];%s=0
B = [0 B];%d=1
Mi = idpoly(1,B,[],[],A2);
Mi.Structure.b.Free = [zeros(1,1) 1];
z_pw = iddata(y_pw.y,u_pw);
Mba2 = pem(z_pw,Mi)
present(Mba2)
v_hat = resid(Mba2,z_pw);
crosscorrel(v_hat.y,u_pw,lag)

%% Modelling x

x = y - filter(Mba2.b, Mba2.f, u);
x = x(10:end);
%acfpacfnorm(x,lag,conf_int)
crosscorrel(x,u,50)
%%
x_data = iddata(x);
A_x = [1 zeros(1,24)];
C_x = [1 zeros(1,24)];
x_poly = idpoly(A_x,[],C_x);
x_poly.Structure.a.Free = [0 1 1 zeros(1,21) 1];
x_poly.Structure.c.Free = [0 zeros(1,23) 1];
model_x = pem(x_data,x_poly);
x_resid = resid(x_data,model_x);%e
acfpacfnorm(x_resid.y,lag,conf_int)




%% Task C














