function [y,Fr1_updated] = GPI_Model(v, Fr1_previous)
    %% v  % 0-10V,0.001s,1000hz
    
    rd = [-8    -6    -4    -2     0     2     4     6     8];
    eta = [1.0000    1.0000    1.0000    1.0000    1.2437   -0.1103   -0.0787   -0.1742   -0.1767];
    r=[0     1     2     3     4     5];
    n=length(r);
    P=[0.7460    0.1860    0.0650    0.0855    0.0362   -0.1159];

    Ad = zeros(length(rd),1);
    for z = 1:length(rd)
        if rd(z) > 0
            Ad(z) = max(v - rd(z), 0);
        elseif rd(z) < 0
            Ad(z) = min(v - rd(z), 0);
        else
            Ad(z) = v;
        end
    end
    w = eta * Ad;

    Fr1 = zeros(n, 1);
    % 对于每个阈值r(j)，更新Fr1(j)
    for j = 1:n
        if ~isempty(Fr1_previous)
            Fr1(j) = Fr1_previous(j);
        end
        F_inc = w;
        F_dec = w;
        A = F_inc - r(j);
        B = min(F_dec + r(j), Fr1(j));
        Fr1(j) = max(A, B); % Equation (1)
    end
    %% 输出y
    y = P*Fr1;
    Fr1_updated = Fr1;
end
