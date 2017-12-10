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
load tid94.mat
%plot(tvxo94)
%figure(2)
%plot(ptvxo94)

vxo = tvxo94(2161:3841); %model set
%vxo_all = tvxo94(2161:8760); %whole set, actually to 8881 (next year)
plot(vxo)
%vxo = vxo.^(1/2);
%plot(vxo_all)
%% looking at u
u = vxo - mean(vxo);
lag = 50;
conf_int = 0.05;
acfpacfnorm(u,lag,conf_int)

%% prewhitening u

A_u = [1 zeros(1,25)];
C_u = [1 zeros(1,25)];
u_data = iddata(u);
u_poly = idpoly(C_u,[],A_u);%1       5         10        15        20        25
u_poly.Structure.a.Free = [0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
u_poly.Structure.c.Free = [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0];
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
u_poly.Structure.c.Free = [0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0];
model_u = pem(u_data,u_poly);
present(model_u)
u_pw = filter(model_u.a,model_u.c,u);
u_pw = u_pw(30:end);
acfpacfnorm(u_pw,lag,conf_int)
subplot(144)
whitenessTest(u_pw)

%% prewhitening y

y_pw = resid(y_data,model_u);
crosscorrel(u_pw,y_pw.y,lag);




