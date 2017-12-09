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
k = 5; %steps to predict
[Fk_y,Gk_y] = diophantine(model_y.c,model_y.a,k);
yhat_gc = filter(Gk_y,model_y.c,y);
samp_remove = max(length(Gk_y),length(C));
yhat_gc = yhat_gc(samp_remove:length(yhat_gc)); %remove samples
plot(yhat_gc,'r')
hold on
plot(y(samp_remove:end),'b')

%% Check residual of prediction
resid_gc = y(samp_remove:end) - yhat_gc;
acfpacfnorm(resid_gc,lag,conf_int)


