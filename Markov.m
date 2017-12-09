function x = Markov(P,n,states)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
x = zeros(1,n);
x(1) = states(1);
state = 1;
for i=2:n
    random = rand(1);
    j = 1;
    while(random > 0)
        random = random - P(state,j);
        j = j + 1;
    end
    j = j - 1;
    state = j;
    x(i) = states(j);
end
end

