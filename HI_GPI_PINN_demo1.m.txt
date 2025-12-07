%% ======================= PINN 逆迟滞模型 Demo ==========================
clear; clc; close all;
rng(0);

%% ======================= 0. 加载数据 & 构建网络 ==========================
load('dataset_GPI_dl_20251208.mat');
load('modelsPINN_preTrained_20251208.mat')

% pinn = HysteresisInversePINN( ...
%     [1 64 64 64 1], ... 
%     struct( ... 
%         'rd', [-8 -6 -4 -2 0 2 4 6 8], ...
%         'eta',[1 1 1 1 1.2437 -0.1103 -0.0787 -0.1742 -0.1767], ...
%         'r',  [0 1 2 3 4 5], ...
%         'P',  [0.7460 0.1860 0.0650 0.0855 0.0362 -0.1159]), ...
%     struct('lambdaData',1,'lambdaPhys',0,'lambdaMono',0,'lambdaSmooth',0), ...
%     @GPI_forward_dl);
% pinn.setNormalization(dataset.normIn, dataset.normOut);

%% ======================= 训练参数表 ==========================
trainParamList = {};

trainParamList{end+1} = struct( ...
    'name', 'Stage1', 'train_ratio', 0.8, ...
    'epochs', 200, 'lrNet', 1e-3, 'lrPhys', 1e-4, ...
    'trainablePhysParams', {{}}, ...
    'lossWeights', struct('lambdaData',1,'lambdaPhys',0,'lambdaMono',0,'lambdaSmooth',0) );
trainParamList{end+1} = struct( ...
    'name', 'Stage1', 'train_ratio', 0.8, ...
    'epochs', 1200, 'lrNet', 1e-4, 'lrPhys', 1e-4, ...
    'trainablePhysParams', {{}}, ...
    'lossWeights', struct('lambdaData',1,'lambdaPhys',0,'lambdaMono',0,'lambdaSmooth',0) );

trainParamList{end+1} = struct( ...
    'name', 'Stage2', 'train_ratio', 0.8, ...
    'epochs', 100, 'lrNet', 5e-4, 'lrPhys', 1e-4, ...
    'trainablePhysParams', {{}}, ...
    'lossWeights', struct('lambdaData',1,'lambdaPhys',10,'lambdaMono',0,'lambdaSmooth', 0) );

trainParamList{end+1} = struct( ...
    'name', 'Stage3', 'train_ratio', 0.8, ...
    'epochs', 100, 'lrNet', 1e-4, 'lrPhys', 1e-4, ...
    'trainablePhysParams', {{}}, ...
    'lossWeights', struct('lambdaData',1,'lambdaPhys',1,'lambdaMono',1e-3,'lambdaSmooth',1e-4) );

%% ======================= 1. 主训练循环 ==========================
if true
    for indexStage = 1:numel(trainParamList)
        for indexGroup = 1:numel(dataset.groups)
            pinn = trainOneStage(pinn, ...
                dataset.groups(indexGroup), ...
                trainParamList{indexStage}, ...
                dataset.normIn, dataset.normOut);
            pause(1) % 等待画图
        end
        pause(1)
    end
end

trainProcessData.lossHistory = pinn.lossHistory;
pinn.clearLossHistory();
trainProcessData.dataset = dataset;
trainProcessData.trainParamList = trainParamList;
save('trainProcessData_pre1.mat', 'trainProcessData');
save('modelsPINN_preTrained', 'pinn');

%% ================== 训练单阶段 ==================
function pinn = trainOneStage(pinn, group, param, normIn, normOut)
    fprintf("\n=== %s | Group %s ===\n", param.name, group.name);
    if isempty(pinn.lossHistory.iter)
        startIdx = 1;
    else
        startIdx = numel(pinn.lossHistory.iter) + 1;
    end

    % 训练集划分（用归一化后的 u_norm / v_norm）
    Ntr = floor(param.train_ratio * numel(group.u_norm));
    Xdl = dlarray(single(group.u_norm(1:Ntr)));
    Ydl = dlarray(single(group.v_norm(1:Ntr)));

    % 设置损失项权重
    pinn.lossWeights = param.lossWeights;

    % 设置本阶段要训练的物理参数
    pinn.trainablePhysParams = param.trainablePhysParams;

    % 使用同一批数据作为物理点（后面你要改物理点采样再换这里）
    X_phys = Xdl;

    % 目前先让 lrNet = lrPhys = param.lr，后面你想拆再拆
    pinn.train( ...
        Xdl, ...                % X_data
        Ydl, ...                % Y_data
        X_phys, ...             % X_phys
        pinn.physicsFcn, ...    % 正向迟滞模型
        pinn.lossWeights, ...   % 损失权重
        param.epochs, ...       % 轮数
        param.lrNet, ...        % lrNet
        param.lrPhys );         % lrPhys

    endIdx = numel(pinn.lossHistory.iter);
    titleName = char(sprintf("%s | Group %s", param.name, group.name));
    plot_stage_summary(pinn, group, param, normIn, normOut, ...
                       titleName, startIdx:endIdx);
end

function plot_stage_summary(pinn, group, param, normIn, normOut, titleName, idxRange)
%PLOT_STAGE_SUMMARY
% 在一个 figure 中绘制：
%   (1) Loss 曲线（可选 idxRange）
%   (2) v(t) Train/Test 拟合 + MSE
%   (3) 滞回曲线 u-v
%
% group: struct，包含 t, u_raw, v_raw, u_norm, v_norm
% param: 本阶段训练参数结构体（name, train_ratio, epochs, lr, trainablePhysParams, lossWeights）
% normIn / normOut: 归一化信息
% titleName: 字符串，用于 figure 名称
% idxRange: 可选，本阶段 lossHistory 的索引范围（startIdx:endIdx）

    % ===== 打印训练参数 =====
    fprintf('Plot summary for %s: ', param.name);
    fprintf('train_ratio=%.2f, epochs=%d, lrNet=%.2g, lrPhys=%.2g\n', ...
        param.train_ratio, param.epochs, param.lrNet, param.lrPhys);
    if isfield(param,'lossWeights')
        lw = param.lossWeights;
        fprintf('  lossWeights: data=%.2g, phys=%.2g, mono=%.2g, smooth=%.2g\n', ...
            lw.lambdaData, lw.lambdaPhys, lw.lambdaMono, lw.lambdaSmooth);
    end
    if isfield(param,'trainablePhysParams') && ~isempty(param.trainablePhysParams)
        names = strjoin(param.trainablePhysParams, ', ');
        fprintf('  trainablePhysParams: {%s}\n', names);
    else
        fprintf('  trainablePhysParams: {}\n');
    end

    % ===== 基本数据拆分 =====
    t = group.t(:);
    u = group.u_raw(:);
    v = group.v_raw(:);
    N = numel(u);
    Ntr = floor(param.train_ratio * N);

    t_train = t(1:Ntr);
    t_test  = t(Ntr+1:end);
    u_train = u(1:Ntr);
    v_train = v(1:Ntr);
    u_test  = u(Ntr+1:end);
    v_test  = v(Ntr+1:end);

    % ===== 0. 建立 figure =====
    if nargin < 6 || isempty(titleName)
        titleName = param.name;
    end
    figure('Name',[char(titleName) ' Summary']);

    %% ================== 1) Loss 曲线 ==================
    subplot(2,1,1); hold on; box on;

    if ~isempty(pinn.lossHistory.components)
        if nargin < 7 || isempty(idxRange)
            idxRange = 1:numel(pinn.lossHistory.iter);
        end

        % it   = pinn.lossHistory.iter(idxRange);
        it   = idxRange - idxRange(1);
        Lall = pinn.lossHistory.components(:, idxRange);

        Ltotal = Lall(1,:);
        Ldata  = Lall(2,:);
        Lphys  = Lall(3,:);
        Lmono  = Lall(4,:);
        Lsmooth= Lall(5,:);

        plot(it, Ltotal, 'k','LineWidth',1.2);
        plot(it, Ldata,  'b--','LineWidth',1.0);
        plot(it, Lphys,  'r--','LineWidth',1.0);
        plot(it, Lmono,  'g--','LineWidth',1.0);
        plot(it, Lsmooth,'m--','LineWidth',1.0);
        xlabel('Iteration'); ylabel('Loss');
        legend('Total','Data','Phys','Mono','Smooth','Location','best');
        title('Loss History');
        grid on;
    else
        title('Loss History (empty)');
        axis off;
    end

    %% ================== 2) v(t) 拟合 ==================
    % 2.1 归一化 u
    [u_train_norm_row, ~] = BasePINN.normalizeData(u_train.', normIn.mode, normIn);
    [u_test_norm_row,  ~] = BasePINN.normalizeData(u_test.',  normIn.mode, normIn);

    % 2.2 前向预测（归一化空间）
    v_train_pred_norm_dl = pinn.forward(dlarray(single(u_train_norm_row)));
    v_test_pred_norm_dl  = pinn.forward(dlarray(single(u_test_norm_row)));

    % 2.3 反归一化到物理域
    v_train_pred_norm_row = double(extractdata(v_train_pred_norm_dl));
    v_test_pred_norm_row  = double(extractdata(v_test_pred_norm_dl));

    v_train_pred_row = BasePINN.denormalizeData(v_train_pred_norm_row, normOut);
    v_test_pred_row  = BasePINN.denormalizeData(v_test_pred_norm_row,  normOut);

    v_train_pred = v_train_pred_row.';   % N_train x 1
    v_test_pred  = v_test_pred_row.';    % N_test  x 1

    mse_train = mean((v_train_pred - v_train).^2);
    mse_test  = mean((v_test_pred  - v_test).^2);

    subplot(2,1,2); hold on; box on;
    plot(t_train, v_train,      'b','LineWidth',1.2);
    plot(t_train, v_train_pred, 'r--','LineWidth',1.2);
    plot(t_test,  v_test,       'c','LineWidth',1.0);
    plot(t_test,  v_test_pred,  'm--','LineWidth',1.0);
    xlabel('t'); ylabel('v');
    title(sprintf('Fit vs Time (MSE_{train}=%.2e, MSE_{test}=%.2e)', mse_train, mse_test));
    legend('v_{train}','v_{train,pred}','v_{test}','v_{test,pred}','Location','best');
    grid on;

    %% ================== 3) 滞回曲线 u-v ==================
    % % 全序列 u_raw -> v_pred -> u_hat_pred(通过正向GPI)
    % 
    % u_raw = u(:)';  % 1×N
    % v_raw = v(:)';
    % 
    % % 1) 归一化 u_raw
    % [u_norm_all_row, ~] = BasePINN.normalizeData(u_raw, normIn.mode, normIn);
    % 
    % % 2) PINN 逆模型预测 v_pred_norm
    % v_pred_norm_dl  = pinn.forward(dlarray(single(u_norm_all_row)));
    % v_pred_norm_row = double(extractdata(v_pred_norm_dl));
    % 
    % % 3) 反归一化到物理电压 v_pred_phys
    % v_pred_phys_row = BasePINN.denormalizeData(v_pred_norm_row, normOut);
    % v_pred_phys     = v_pred_phys_row';    % N×1
    % 
    % % 4) 正向 GPI：v_pred_phys -> u_hat_pred
    % u_hat_pred_dl = pinn.physicsFcn(dlarray(single(v_pred_phys')), pinn.paramsPhys);
    % u_hat_pred    = double(extractdata(u_hat_pred_dl))';
    % 
    % subplot(3,1,3); hold on; box on;
    % plot(v_raw, u_raw,       'b.', 'DisplayName','True');
    % plot(v_pred_phys, u_hat_pred, 'r.', 'DisplayName','Pred (inverse→forward)');
    % xlabel('v'); ylabel('u');
    % title('Hysteresis Loop');
    % legend('Location','best');
    % grid on;

end
