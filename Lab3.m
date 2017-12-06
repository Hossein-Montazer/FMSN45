load tar2.dat
load thx.dat
tar2_data = iddata(tar2);
subplot(311)
plot(tar2)
subplot(312)
plot(thx)
na = 2;
nb = 0;
nk = 0;
model = [na]; %[na,nb,nk]
lambda = 0.99;
[Aest,yhat,covAest,yprev]= rarx(tar2_data,model,'ff',lambda);
subplot(313)
plot(Aest)
%%
n = 100;
lambda_line = linspace(0.85,1,n);
ls2 = zeros(n,1);
for i=1:length(lambda_line)
    [Aest,yhat,CovAest,trash] = rarx(tar2,[2],'ff',lambda_line(i));
    ls2(i) = sum((tar2-yhat).^2);
end
figure(2)
plot(lambda_line,ls2)














%% Task 3.4
load svedala94.mat
plot(svedala94)
Diff = [1 0 0 0 0 0 -1];
y_diff = filter(Diff,1,svedala94);
T = linspace(datenum(1994,1,1),datenum(1994,12,31),length(svedala94));
subplot(211)
plot(T,svedala94);
datetick('x');
subplot(212)
plot(T,y_diff);
datetick('x');

%% Q.6
th = armax(y_diff,[2 2]);
th_winter = armax(y_diff(1:540),[2 2]);
th_summer = armax(y_diff(907:1458),[2 2]);

%% Q.7
th0 = [th_winter.A(2:end) th_winter.C(2:end)];
[thr,yhat] = rarmax(y_diff,[2 2],'ff',0.99,th0);

subplot(311)
plot(T,svedala94)
datetick('x');

subplot(312)
plot(thr(:,1:2))
hold on
plot(repmat(th_winter.A(2:end),[length(thr) 1]), 'b:');
plot(repmat(th_summer.A(2:end),[length(thr) 1]), 'r:');
axis tight
hold off

subplot(313)
plot(thr(:,3:end))
hold on
plot(repmat(th_winter.C(2:end),[length(thr) 1]), 'b:');
plot(repmat(th_summer.C(2:end),[length(thr) 1]), 'r:');
axis tight
hold off




%% Task 3.5
load svedala94.mat
%y = svedala94(600:850);
y = svedala94;
%y = y - mean(y);
y = y-y(1);
t = (1:length(y))';
U = [sin(2*pi*t/6) cos(2*pi*t/6)];
Z = iddata(y,U);
model = [3 [1 1] 4 [0 0]]; %[na [nb_1 nb_2] nc [nk1_ nk_2]]
thx = armax(Z,model);
plot(y)
hold on
asdf = U*cell2mat(thx.b)';
%sincos = zeros(1,length(asdf));
%for i=1:length(asdf)
%    sincos(i) = asdf(i,1)+asdf(i,2);
%end
plot(asdf)

%% Q.9,10

U = [sin(2*pi*t/6) cos(2*pi*t/6) ones(size(t))];
Z = iddata(y,U);
m0 = [thx.A(2:end) thx.B 0 thx.C(2:end)];
m0 = cell2mat(m0);
Re = diag([0 0 0 0 0 1 0 0 0 0]);
model = [3 [1 1 1] 4 0 [0 0 0] [1 1 1]];
[thr,yhat] = rpem(Z,model,'kf',Re,m0);
plot(thr)
%%
m = thr(:,6);
a = thr(end,4);
b = thr(end,5);
y_mean = m + a*U(:,1)+b*U(:,2);
y_mean = [0;y_mean(1:end-1)];
plot(y)
hold on
plot(y_mean)
hold off






