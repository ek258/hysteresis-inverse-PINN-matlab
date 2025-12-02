%% demo_hysteresis_inverse_PINN_staged_plot.m
% 基于 BasePINN + HysteresisInversePINN 的分阶段训练 demo
% - 每阶段训练后画 train/test 拟合 + loss
% - 训练结束后画总的拟合 + loss + 滞回曲线

clear; clc; close all;
rng(0);

%% ===== 0. 一些训练超参数 =====
% 时间与采样
dt      = 1e-3;
T_end   = 1.2;
t       = (0:dt:T_end-dt).';
N       = numel(t);

train_ratio = 0.8;
N_train = floor(N * train_ratio);
t_train = t(1:N_train);
t_test  = t(N_train+1:end);

% 各阶段下采样 / 物理点抽样步长（原始设定）
data_stride_stage1 = 5;
data_stride_stage2 = 2;
phys_stride_stage2 = 5;
phys_stride_stage3 = 5;

% 分阶段训练轮数 / 学习率
epochs1 = 2000; lr1 = 1e-3;   % Data only
epochs2 = 3000; lr2 = 1e-3;   % Data + Phys
epochs3 = 1000; lr3 = 2e-4;   % Full loss

%% ===== 1. 生成全序列数据 (GPI 正向迟滞) =====
% 逆模型输入：u_all（位移），输出：v_all（电压）
v_all = 5 + 4*sin(2*pi*1*t) + 1.5*sin(2*pi*3*t);
v_all = max(0, min(10, v_all));   % 限幅 [0,10]

u_all = zeros(size(v_all));
Fr = [];
for k = 1:N
    [u_all(k), Fr] = GPI_Model(v_all(k), Fr);   % 这里调用你已有的 GPI_Model
end

u_train_full = u_all(1:N_train);
v_train_full = v_all(1:N_train);
u_test_full  = u_all(N_train+1:end);
v_test_full  = v_all(N_train+1:end);

%% ===== 1.2 归一化 (z-score) =====
% 注意：网络在“归一化空间”里工作：u_norm -> v_norm
%       画图/MSE 时再反归一化回物理量

% 训练集归一化
[u_train_norm_row, normIn]  = BasePINN.normalizeData(u_train_full.', 'zscore');
[v_train_norm_row, normOut] = BasePINN.normalizeData(v_train_full.', 'zscore');

% 测试集用同一个 normInfo 做变换（保持一致）
[u_test_norm_row, ~]  = BasePINN.normalizeData(u_test_full.',  'zscore', normIn);
[v_test_norm_row, ~]  = BasePINN.normalizeData(v_test_full.',  'zscore', normOut);

% 方便后面直接用的 dlarray 形式（归一化空间）
X_test_dl = dlarray(single(u_test_norm_row));
Y_test_dl = dlarray(single(v_test_norm_row));

%% ===== 2. 正向迟滞参数（给 physicsFcn 用） =====
physParam.rd  = [-8 -6 -4 -2 0 2 4 6 8];
physParam.eta = [1 1 1 1 1.2437 -0.1103 -0.0787 -0.1742 -0.1767];
physParam.r   = [0 1 2 3 4 5];
physParam.P   = [0.7460 0.1860 0.0650 0.0855 0.0362 -0.1159];

%% ===== 3. 定义网络 =====
layers = [1 64 64 1];
dummyWeights = struct('lambdaData',1,'lambdaPhys',0,'lambdaMono',0,'lambdaSmooth',0);
pinn = HysteresisInversePINN(layers, physParam, dummyWeights, @physicsFcn_GPI_forward);

% 设置归一化参数（供 computeLoss / 其他方法内部使用）
% 假定你在 HysteresisInversePINN 里已经实现了 setNormalization / applyDenormOut 等
pinn.setNormalization(normIn, normOut);

%% 工具：记录各阶段 lossHistory 范围
stageRanges = struct('name',{},'idx',{});

%% ===== 4. Stage 1: Data only + 强下采样 =====
disp("===== Stage 1: Data only =====");

idx_s1 = 1:data_stride_stage1:N_train;

% 训练数据在归一化空间：u_norm -> v_norm
X_s1_dl = dlarray(single(u_train_norm_row(idx_s1)));
Y_s1_dl = dlarray(single(v_train_norm_row(idx_s1)));

lossW1.lambdaData   = 1;
lossW1.lambdaPhys   = 0;
lossW1.lambdaMono   = 0;
lossW1.lambdaSmooth = 0;

prevLen = numel(pinn.lossHistory.total);
pinn.train(X_s1_dl, Y_s1_dl, X_s1_dl, ...
           @physicsFcn_GPI_forward, physParam, lossW1, ...
           epochs1, lr1);
newLen = numel(pinn.lossHistory.total);
stageRanges(end+1).name = 'Stage 1'; 
stageRanges(end).idx    = prevLen+1:newLen;

% --- Stage 1 绘图（内部会自动做归一化/反归一化） ---
plot_stage_fit(pinn, 1, t_train, u_train_full, v_train_full, ...
               t_test, u_test_full, v_test_full, ...
               normIn, normOut);
plot_stage_loss(pinn, 1, stageRanges(end).idx);

%% ===== 5. Stage 2: Data + Physics，中等下采样 + 动态物理点抽样 =====
disp("===== Stage 2: Data + Physics =====");

idx_s2 = 1:data_stride_stage2:N_train;

X_s2_dl = dlarray(single(u_train_norm_row(idx_s2)));
Y_s2_dl = dlarray(single(v_train_norm_row(idx_s2)));

% 原来是固定步长抽物理点，这里改成“残差+随机”的动态抽样
% 物理点数量保持和原来大致一致：ceil(N_s2 / phys_stride_stage2)
N_s2     = numel(idx_s2);
N_phys2  = ceil(N_s2 / phys_stride_stage2);
X_phys_s2_dl = samplePhysicsPoints(pinn, ...
    u_train_full, u_train_norm_row, ...
    physParam, N_phys2, 'hybrid');  % 'random' | 'residual' | 'hybrid'

lossW2.lambdaData   = 1;
lossW2.lambdaPhys   = 10;
lossW2.lambdaMono   = 0;
lossW2.lambdaSmooth = 0;

prevLen = numel(pinn.lossHistory.total);
pinn.train(X_s2_dl, Y_s2_dl, X_phys_s2_dl, ...
           @physicsFcn_GPI_forward, physParam, lossW2, ...
           epochs2, lr2);
newLen = numel(pinn.lossHistory.total);
stageRanges(end+1).name = 'Stage 2'; 
stageRanges(end).idx    = prevLen+1:newLen;

% --- Stage 2 绘图 ---
plot_stage_fit(pinn, 2, t_train, u_train_full, v_train_full, ...
               t_test, u_test_full, v_test_full, ...
               normIn, normOut);
plot_stage_loss(pinn, 2, stageRanges(end).idx);

%% ===== 6. Stage 3: Data + Physics + Mono + Smooth =====
disp("===== Stage 3: Full loss =====");

idx_s3 = idx_s2;          % 这里用和 Stage2 相同的下采样

X_s3_dl = dlarray(single(u_train_norm_row(idx_s3)));   % 输入 u_norm
Y_s3_dl = dlarray(single(v_train_norm_row(idx_s3)));   % 输出 v_norm

N_s3     = numel(idx_s3);
N_phys3  = ceil(N_s3 / phys_stride_stage3);
X_phys_s3_dl = samplePhysicsPoints(pinn, ...
    u_train_full, u_train_norm_row, ...
    physParam, N_phys3, 'hybrid');

lossW3.lambdaData   = 5;
lossW3.lambdaPhys   = 5;
lossW3.lambdaMono   = 0.1;
lossW3.lambdaSmooth = 0.001;

prevLen = numel(pinn.lossHistory.total);
pinn.train(X_s3_dl, Y_s3_dl, X_phys_s3_dl, ...
           @physicsFcn_GPI_forward, physParam, lossW3, ...
           epochs3, lr3);
newLen = numel(pinn.lossHistory.total);
stageRanges(end+1).name = 'Stage 3'; 
stageRanges(end).idx    = prevLen+1:newLen;

% --- Stage 3 绘图 ---
plot_stage_fit(pinn, 3, t_train, u_train_full, v_train_full, ...
               t_test, u_test_full, v_test_full, ...
               normIn, normOut);
plot_stage_loss(pinn, 3, stageRanges(end).idx);

%% ===== 7. 训练结束后：总拟合 + 总 loss + 滞回曲线 =====
disp("===== Final plots =====");

% ---- 总拟合：train + test ----
% 1) 把全量 u_train / u_test 变成归一化空间
[u_train_norm_full_row, ~] = BasePINN.normalizeData(u_train_full.', normIn.mode, normIn);
[u_test_norm_full_row,  ~] = BasePINN.normalizeData(u_test_full.',  normIn.mode, normIn);

% 2) 网络前向：u_norm -> v_norm
v_train_pred_norm_dl = pinn.forward(dlarray(single(u_train_norm_full_row)));
v_test_pred_norm_dl  = pinn.forward(dlarray(single(u_test_norm_full_row)));

v_train_pred_norm_row = double(extractdata(v_train_pred_norm_dl));
v_test_pred_norm_row  = double(extractdata(v_test_pred_norm_dl));

% 3) 反归一化到物理电压
v_train_pred_row = BasePINN.denormalizeData(v_train_pred_norm_row, normOut);
v_test_pred_row  = BasePINN.denormalizeData(v_test_pred_norm_row,  normOut);

v_train_pred = v_train_pred_row.';   % N_train x 1
v_test_pred  = v_test_pred_row.';    % N_test x 1

mse_train = mean((v_train_pred - v_train_full).^2);
mse_test  = mean((v_test_pred  - v_test_full).^2);

figure('Name','Final Fit (Train + Test)');
subplot(2,1,1); hold on; box on;
plot(t_train, v_train_full, 'b', 'LineWidth',1.2);
plot(t_train, v_train_pred, 'r--','LineWidth',1.2);
xlabel('t [s]'); ylabel('Voltage v [V]');
title(sprintf('Train fit (MSE=%.2e)', mse_train));
legend('True v','PINN v_{pred}','Location','best');

subplot(2,1,2); hold on; box on;
plot(t_test, v_test_full, 'b', 'LineWidth',1.2);
plot(t_test, v_test_pred, 'r--','LineWidth',1.2);
xlabel('t [s]'); ylabel('Voltage v [V]');
title(sprintf('Test fit (MSE=%.2e)', mse_test));
legend('True v','PINN v_{pred}','Location','best');

% ---- 总 loss 曲线 ----
if ~isempty(pinn.lossHistory.components)
    it_all = pinn.lossHistory.iter;
    Lall   = pinn.lossHistory.components;   % 5 x K: [total,data,phys,mono,smooth]
    Ltotal = Lall(1,:);
    Ldata  = Lall(2,:);
    Lphys  = Lall(3,:);
    Lmono  = Lall(4,:);
    Lsmooth= Lall(5,:);

    figure('Name','Total Loss History');
    hold on; box on;
    plot(it_all, Ltotal, 'k','LineWidth',1.2);
    plot(it_all, Ldata,  'b--','LineWidth',1.0);
    plot(it_all, Lphys,  'r--','LineWidth',1.0);
    plot(it_all, Lmono,  'g--','LineWidth',1.0);
    plot(it_all, Lsmooth,'m--','LineWidth',1.0);
    xlabel('Iteration'); ylabel('Loss');
    legend('Total','Data','Phys','Mono','Smooth','Location','best');
    title('Overall Loss History');
    grid on;
end

% ---- 滞回曲线：真实 (v_all,u_all) vs PINN 逆 + 正向 GPI ----
% 1) 全序列 u_all -> 归一化 u_norm_all
[u_all_norm_row, ~] = BasePINN.normalizeData(u_all.', normIn.mode, normIn);

% 2) 逆模型：u_norm_all -> v_norm_all
v_pred_all_norm_dl  = pinn.forward(dlarray(single(u_all_norm_row)));
v_pred_all_norm_row = double(extractdata(v_pred_all_norm_dl));

% 3) 反归一化到物理电压
v_pred_all_row = BasePINN.denormalizeData(v_pred_all_norm_row, normOut);
v_pred_all     = v_pred_all_row.';   % N x 1

% 4) 正向 GPI：v_pred_all -> u_hat_all
u_hat_all_dl = physicsFcn_GPI_forward(dlarray(single(v_pred_all.')), physParam);
u_hat_all    = double(extractdata(u_hat_all_dl)).';

figure('Name','Hysteresis Loop');
hold on; box on;
plot(v_all,       u_all,    'b.', 'DisplayName','True hysteresis');
plot(v_pred_all,  u_hat_all,'r.', 'DisplayName','PINN inverse -> forward');
xlabel('Voltage v'); ylabel('Displacement u');
legend('Location','best');
title('Hysteresis loop comparison');
grid on;

function record_loss(pinn, params, Xd, Yd, Xp, physFcn, physParam, W)
    Lvec = dlfeval(@BasePINN.evalWrapper, pinn, params, ...
                   Xd, Yd, Xp, physFcn, physParam, W);
    pinn.lossHistory.iter(end+1)  = pinn.iteration;
    pinn.lossHistory.total(end+1) = double(Lvec(1));
    if isempty(pinn.lossHistory.components)
        pinn.lossHistory.components = double(Lvec(:));
    else
        pinn.lossHistory.components(:, end+1) = double(Lvec(:));
    end
end

%% ================== 辅助绘图函数 ==================
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
