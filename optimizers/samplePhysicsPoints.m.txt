%% ================== 动态物理点采样函数 ==================
function X_phys_dl = samplePhysicsPoints(pinn, ...
    u_train_full, u_train_norm_row, ...
    physParam, N_phys, mode)
% 动态采样物理点
%  - pinn: 已训练到当前阶段的 PINN 对象
%  - u_train_full      : N_train x 1 (物理量)
%  - u_train_norm_row  : 1 x N_train (归一化空间)
%  - N_phys            : 本次要选多少个 collocation 点
%  - mode              : 'random' | 'residual' | 'hybrid'

    if nargin < 6 || isempty(mode)
        mode = 'hybrid';
    end

    N_train = numel(u_train_full);

    switch lower(mode)
        case 'random'
            idx_phys = randperm(N_train, min(N_phys, N_train));

        case 'residual'
            % 使用当前网络计算全局物理残差 |H(v_pred) - u|
            [~, idx_phys] = residualTopK(pinn, u_train_full, u_train_norm_row, ...
                                         physParam, N_phys);

        otherwise % 'hybrid'：一半随机，一半残差大的
            N1 = floor(N_phys/2);
            N2 = N_phys - N1;

            idx_rand = randperm(N_train, min(N1, N_train));
            [~, idx_top] = residualTopK(pinn, u_train_full, u_train_norm_row, ...
                                        physParam, min(N2, N_train));

            idx_phys = unique([idx_rand(:); idx_top(:)]).';
    end

    % 取出对应的归一化 u_norm 作为 X_phys
    u_phys_norm_seg = u_train_norm_row(:, idx_phys);  % 1 x N_phys
    X_phys_dl = dlarray(single(u_phys_norm_seg));
end

function [e_sorted, idx_top] = residualTopK(pinn, u_train_full, u_train_norm_row, ...
                                            physParam, K)
    % 1) 在所有训练点上预测 v（归一化空间）
    u_norm_dl = dlarray(single(u_train_norm_row));
    v_norm_dl = pinn.forward(u_norm_dl);
    v_norm    = double(extractdata(v_norm_dl));   % 1 x N

    % 2) 反归一化到物理空间 v_phys
    % 这里假定 pinn 内部有 applyDenormOut，如果没有可以改成:
    %   v_phys_row = BasePINN.denormalizeData(v_norm, pinn.normOut);
    v_phys_row = pinn.applyDenormOut(v_norm);
    v_phys     = v_phys_row.';                    % N x 1

    % 3) 正向迟滞模型
    u_hat_dl = physicsFcn_GPI_forward(dlarray(single(v_phys.')), physParam);
    u_hat    = double(extractdata(u_hat_dl)).';   % N x 1

    % 4) 残差
    e = abs(u_hat - u_train_full);               % N x 1
    [e_sorted, idx_sorted] = sort(e, 'descend');
    K = min(K, numel(e_sorted));
    idx_top = idx_sorted(1:K);
end