%% 

clear; clc; close all;
rng(0);

%% ======================= 0. 加载数据 & 构建网络 ==========================
load('dataset_GPI_dl.mat','dataset');

pinn = HysteresisInversePINN(...
    [1 64 64 1], ... 
    struct( ... 
        'rd', [-8 -6 -4 -2 0 2 4 6 8], ...
        'eta', [1 1 1 1 1.2437 -0.1103 -0.0787 -0.1742 -0.1767], ...
        'r', [0 1 2 3 4 5], ...
        'P', [0.7460 0.1860 0.0650 0.0855 0.0362 -0.1159]), ...
    struct('lambdaData', 1, 'lambdaPhys', 0, 'lambdaMono', 0, 'lambdaSmooth', 0), ...
    @GPI_forward_dl); 
pinn.setNormalization(dataset.normIn, dataset.normOut);

trainParamList = {};
trainParamList{end+1} = struct('name', 'Stage1', 'train_ratio', 0.8, ...
    'epochs', 2000, 'lr', 1e-3, ...
    'lossWeights', struct('lambdaData', 1, 'lambdaPhys', 0, ...
        'lambdaMono', 0, 'lambdaSmooth', 0));
trainParamList{end+1} = struct('name', 'Stage2', 'train_ratio', 0.8, ...
    'epochs', 2000, 'lr', 1e-3, ...
    'lossWeights', struct('lambdaData', 1, 'lambdaPhys', 0, ...
        'lambdaMono', 0, 'lambdaSmooth', 0));
trainParamList{end+1} = struct('name', "Stage3", 'train_ratio', 0.8, ...
    'epochs', 2000, 'lr', 1e-3, ...
    'lossWeights', struct('lambdaData', 1, 'lambdaPhys', 0, ...
        'lambdaMono', 0, 'lambdaSmooth', 0));

%% ======================= 1. 主训练循环 ==========================
if true
    for stageIndex = 1:numel(trainParamList)
        for groupIndex = 1:numel(dataset.groups)
            trainOneStage(pinn, dataset.groups(groupIndex), trainParamList(stageIndex));
            % (待完成)绘制loss曲线、拟合曲线、预测曲线、滞回曲线，在一个图窗中
        end
    end
end

function pinn = trainOneStage(pinn, group, param)
    fprintf("\n===== Training %s =====\n", param.name);

    Ntr = floor(param.train_ratio * numel(group.u_norm));
    Xdl = dlarray(single(group.u_norm(1:Ntr)));
    Ydl = dlarray(single(group.v_norm(1:Ntr)));

    pinn.lossWeights = param.lambda;
    pinn.train(Xdl, Ydl, Xdl, ...
        pinn.physicsFcn, pinn.physParam, ...
        param.lambda, param.epochs, param.lr );

end

%% ================== 辅助绘图函数 ==================
% 以下函数你可以随意修改和增删
function plot_stage_fit(pinn, stageId, t_train, u_train, v_train, ...
                        t_test,  u_test,  v_test, ...
                        normIn, normOut)
    % 1) 物理 u -> 归一化 u_norm
    [u_train_norm_row, ~] = BasePINN.normalizeData(u_train.', normIn.mode, normIn);
    [u_test_norm_row,  ~] = BasePINN.normalizeData(u_test.',  normIn.mode, normIn);

    % 2) 前向预测（归一化空间）
    v_train_pred_norm_dl = pinn.forward(dlarray(single(u_train_norm_row)));
    v_test_pred_norm_dl  = pinn.forward(dlarray(single(u_test_norm_row)));

    v_train_pred_norm_row = double(extractdata(v_train_pred_norm_dl));
    v_test_pred_norm_row  = double(extractdata(v_test_pred_norm_dl));

    % 3) 反归一化回物理电压
    v_train_pred_row = BasePINN.denormalizeData(v_train_pred_norm_row, normOut);
    v_test_pred_row  = BasePINN.denormalizeData(v_test_pred_norm_row,  normOut);

    v_train_pred = v_train_pred_row.';  % N_train x 1
    v_test_pred  = v_test_pred_row.';   % N_test x 1

    mse_train = mean((v_train_pred - v_train).^2);
    mse_test  = mean((v_test_pred  - v_test).^2);

    figure('Name',sprintf('Stage %d Fit', stageId));
    subplot(2,1,1); hold on; box on;
    plot(t_train, v_train, 'b','LineWidth',1.2);
    plot(t_train, v_train_pred,'r--','LineWidth',1.2);
    xlabel('t [s]'); ylabel('Voltage v [V]');
    title(sprintf('Stage %d Train fit (MSE=%.2e)', stageId, mse_train));
    legend('True v','PINN v_{pred}','Location','best');

    subplot(2,1,2); hold on; box on;
    plot(t_test, v_test, 'b','LineWidth',1.2);
    plot(t_test, v_test_pred,'r--','LineWidth',1.2);
    xlabel('t [s]'); ylabel('Voltage v [V]');
    title(sprintf('Stage %d Test fit (MSE=%.2e)', stageId, mse_test));
    legend('True v','PINN v_{pred}','Location','best');
end

function plot_stage_loss(pinn, stageId, idxRange)
    if isempty(pinn.lossHistory.components)
        return;
    end
    it   = pinn.lossHistory.iter(idxRange);
    Lall = pinn.lossHistory.components(:, idxRange);
    Ltotal = Lall(1,:);
    Ldata  = Lall(2,:);
    Lphys  = Lall(3,:);
    Lmono  = Lall(4,:);
    Lsmooth= Lall(5,:);

    figure('Name',sprintf('Stage %d Loss', stageId));
    hold on; box on;
    plot(it, Ltotal, 'k','LineWidth',1.2);
    plot(it, Ldata,  'b--','LineWidth',1.0);
    plot(it, Lphys,  'r--','LineWidth',1.0);
    plot(it, Lmono,  'g--','LineWidth',1.0);
    plot(it, Lsmooth,'m--','LineWidth',1.0);
    xlabel('Iteration'); ylabel('Loss');
    legend('Total','Data','Phys','Mono','Smooth','Location','best');
    title(sprintf('Stage %d Loss History', stageId));
    grid on;
end