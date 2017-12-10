function [ output_args ] = crosscorrel(u,v,lag)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    %subplot(144)
    stem(-lag:lag,crosscorr(u,v,lag));
    title('Cross correlation function'), xlabel('lag')
    hold on
    plot(-lag:lag, 2/sqrt(length(u))*ones(1,2*lag+1),'--')
    plot(-lag:lag, -2/sqrt(length(u))*ones(1,2*lag+1),'--')
    hold off

end

