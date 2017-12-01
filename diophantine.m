function [Fk,Gk] = diophantine(C,A,k)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[CS,AS] = equalLength(C,A);
[Fk,Gk] = deconv(conv([1 zeros(1,k-1)],CS),AS);

end

