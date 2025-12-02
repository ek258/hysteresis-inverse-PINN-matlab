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

% 各阶段下采样 / 物理点抽样
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
u_test       = u_all(N_train+1:end);
v_test       = v_all(N_train+1:end);

X_test_dl = dlarray(single(u_test'));
Y_test_dl = dlarray(single(v_test'));

%% ===== 2. 正向迟滞参数（给 physicsFcn 用） =====
physParam.rd  = [-8 -6 -4 -2 0 2 4 6 8];
physParam.eta = [1 1 1 1 1.2437 -0.1103 -0.0787 -0.1742 -0.1767];
physParam.r   = [0 1 2 3 4 5];
physParam.P   = [0.7460 0.1860 0.0650 0.0855 0.0362 -0.1159];

%% ===== 3. 定义网络 =====
layers = [1 64 64 1];
dummyWeights = struct('lambdaData',1,'lambdaPhys',0,'lambdaMono',0,'lambdaSmooth',0);
pinn = HysteresisInversePINN(layers, physParam, dummyWeights, @physicsFcn_GPI_forward);

%% 工具：记录各阶段 lossHistory 范围
stageRanges = struct('name',{},'idx',{});

%% ===== 4. Stage 1: Data only + 强下采样 =====
disp("===== Stage 1: Data only =====");

idx_s1 = 1:data_stride_stage1:N_train;
u_s1 = u_train_full(idx_s1);
v_s1 = v_train_full(idx_s1);

X_s1_dl = dlarray(single(u_s1'));
Y_s1_dl = dlarray(single(v_s1'));

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

% --- Stage 1 绘图 ---
plot_stage_fit(pinn, 1, t_train, u_train_full, v_train_full, ...
               t_test, u_test, v_test);
plot_stage_loss(pinn, 1, stageRanges(end).idx);

%% ===== 5. Stage 2: Data + Physics，中等下采样 + 物理点抽样 =====
disp("===== Stage 2: Data + Physics =====");

idx_s2 = 1:data_stride_stage2:N_train;
u_s2 = u_train_full(idx_s2);
v_s2 = v_train_full(idx_s2);

X_s2_dl = dlarray(single(u_s2'));
Y_s2_dl = dlarray(single(v_s2'));

idx_phys2 = 1:phys_stride_stage2:numel(u_s2);
X_phys_s2_dl = X_s2_dl(:, idx_phys2);

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
               t_test, u_test, v_test);
plot_stage_loss(pinn, 2, stageRanges(end).idx);

%% ===== 6. Stage 3: Data + Physics + Mono + Smooth =====
disp("===== Stage 3: Full loss =====");

idx_s3 = idx_s2;          % 这里用和 Stage2 相同的下采样
u_s3   = u_s2;
v_s3   = v_s2;

X_s3_dl = dlarray(single(u_s3'));   % 输入 u
Y_s3_dl = dlarray(single(v_s3'));   % 输出 v  

idx_phys3     = 1:phys_stride_stage3:numel(u_s3);
X_phys_s3_dl  = X_s3_dl(:, idx_phys3);

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
               t_test, u_test, v_test);
plot_stage_loss(pinn, 3, stageRanges(end).idx);

%% ===== 7. 训练结束后：总拟合 + 总 loss + 滞回曲线 =====
disp("===== Final plots =====");

% ---- 总拟合：train + test ----
v_train_pred_dl = pinn.forward(dlarray(single(u_train_full')));
v_train_pred = double(extractdata(v_train_pred_dl)).';

v_test_pred_dl = pinn.forward(X_test_dl);
v_test_pred = double(extractdata(v_test_pred_dl)).';

mse_train = mean((v_train_pred - v_train_full).^2);
mse_test  = mean((v_test_pred  - v_test).^2);

figure('Name','Final Fit (Train + Test)');
subplot(2,1,1); hold on; box on;
plot(t_train, v_train_full, 'b', 'LineWidth',1.2);
plot(t_train, v_train_pred, 'r--','LineWidth',1.2);
xlabel('t [s]'); ylabel('Voltage v [V]');
title(sprintf('Train fit (MSE=%.2e)', mse_train));
legend('True v','PINN v_{pred}','Location','best');

subplot(2,1,2); hold on; box on;
plot(t_test, v_test, 'b', 'LineWidth',1.2);
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
% 先用逆模型预测 v_pred_all，再过 physicsFcn 得到 u_hat_all
v_pred_all_dl = pinn.forward(dlarray(single(u_all')));
v_pred_all = double(extractdata(v_pred_all_dl)).';

u_hat_all_dl = physicsFcn_GPI_forward(dlarray(single(v_pred_all')), physParam);
u_hat_all = double(extractdata(u_hat_all_dl)).';

figure('Name','Hysteresis Loop');
hold on; box on;
plot(v_all, u_all, 'b.', 'DisplayName','True hysteresis');
plot(v_pred_all, u_hat_all, 'r.', 'DisplayName','PINN inverse -> forward');
xlabel('Voltage v'); ylabel('Displacement u');
legend('Location','best');
title('Hysteresis loop comparison');
grid on;

%% ================== 辅助绘图函数 ==================
function plot_stage_fit(pinn, stageId, t_train, u_train, v_train, ...
                        t_test, u_test, v_test)

    % 预测 train / test
    v_train_pred = double(extractdata( ...
        pinn.forward(dlarray(single(u_train'))))).';
    v_test_pred = double(extractdata( ...
        pinn.forward(dlarray(single(u_test'))))).';

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
