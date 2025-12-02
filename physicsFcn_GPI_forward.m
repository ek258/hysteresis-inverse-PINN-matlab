%% ===== 正向迟滞算子（GPI）：dlarray 版本 =====
function u_hat = physicsFcn_GPI_forward(v_pred, physParam)
    rd  = physParam.rd;
    eta = physParam.eta;
    r   = physParam.r;
    P   = physParam.P;

    if ~isa(v_pred, 'dlarray')
        v_pred = dlarray(single(v_pred));
    else
        if ~isa(extractdata(v_pred), 'single')
            v_pred = dlarray(single(extractdata(v_pred)));
        end
    end

    N  = size(v_pred, 2);
    nR = numel(r);
    nD = numel(rd);

    Fr1_prev = zeros(nR, 1, 'single');
    u_hat_data = zeros(1, N, 'single');

    for k = 1:N
        v = v_pred(1,k);

        % Dead-zone
        Ad = zeros(nD,1,'single');
        for z = 1:nD
            if rd(z) > 0
                Ad(z) = max(v - rd(z), 0);
            elseif rd(z) < 0
                Ad(z) = min(v - rd(z), 0);
            else
                Ad(z) = v;
            end
        end
        w = eta * Ad;

        % Play operator
        Fr1 = zeros(nR,1,'single');
        for j = 1:nR
            A = w - r(j);
            B = min(w + r(j), Fr1_prev(j));
            Fr1(j) = max(A, B);
        end

        Fr1_prev = Fr1;
        u_hat_data(1,k) = P * Fr1;
    end

    u_hat = dlarray(u_hat_data);
end