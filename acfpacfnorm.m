function [ output_args ] = acfpacfnorm(u, lag, conf_int)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

subplot(141)
acf(u,lag,conf_int,1,[],0);
title('Auto-correlation function'), xlabel('lag')
subplot(142)
pacf(u,lag,conf_int,1,0);
title('Partial auto-correlation function'), xlabel('lag')
subplot(143)
normplot(u)
%plotNTdist(u)
end

