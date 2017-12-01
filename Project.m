%% choose data
load utempAva_9395.dat %full zeros before 6803
temp = utempAva_9395(:,3); %90*24=2160  70*24=1680 Yr1994 starts at 8761
%plot(temp) %look at whole data
temp = utempAva_9395(10921:12601,3); %start of april -> mid-june
plot(temp)

%% remove linear trend
temp = temp - mean(trend);
plot(temp)






