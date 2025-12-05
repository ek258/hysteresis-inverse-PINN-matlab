%% ===================================================================
% generate_dataset_GPI_dl_structured.m
% 使用 GPI_forward_dl (可学习参数) 生成多组迟滞数据
% 加入 t, dt, T_end，修复 PRBS 警告，全局归一化
%
% 输出：dataset_GPI_dl.mat
% ====================================================================

clear; clc; close all;
rng(0);
addpath('.');

%% ====================== 1. 时间参数（全部保存） ====================
dt    = 1e-3;
T_end = 1.2;
t     = (0:dt:T_end-dt).';
N     = numel(t);

%% ====================== 2. 激励列表 ================================
excitationList = {};

excitationList{end+1} = @(t) 5 + 4*sin(2*pi*1*t);      % 低频正弦
excitationList{end+1} = @(t) 5 + 3*sin(2*pi*20*t);     % 高频正弦
excitationList{end+1} = @(t) 5 + 3*sin(2*pi*1*t) + ...
                              2*sin(2*pi*5*t) + ...
                              1*sin(2*pi*15*t);        % 多频组合
excitationList{end+1} = @(t) 5 + 4*prbs_signal(N);

% —— Chirp
excitationList{end+1} = @(t) 5 + 4*chirp(t,0.1,t(end),50);

numGroups = numel(excitationList);



%% ====================== 3. GPI_forward_dl 参数 =====================
theta = struct();
theta.rd  = [-8 -6 -4 -2 0 2 4 6 8];
theta.eta = [1 1 1 1 1.2437 -0.1103 -0.0787 -0.1742 -0.1767];
theta.r   = [0 1 2 3 4 5];
theta.P   = [0.7460 0.1860 0.0650 0.0855 0.0362 -0.1159];

%% ====================== 4. 逐组生成数据 ===========================
dataset = struct();
dataset.dt = dt;
dataset.T_end = T_end;
dataset.numGroups = numGroups;

all_v = [];
all_u = [];

fprintf("Generating %d excitation groups...\n", numGroups);

for k = 1:numGroups
    fprintf("  Group %d / %d...\n", k, numGroups);

    % ——原始激励——
    v = excitationList{k}(t);
    v = max(0, min(10, v));   % 限幅
    
    % ——使用 GPI_forward_dl —— 
    Fr0 = [];
    [u_dl, ~] = GPI_forward_dl(v.', theta, Fr0);
    u = double(extractdata(u_dl)).';

    % ——保存原始数据——
    dataset.groups(k).name = sprintf("excitation_%02d", k);
    dataset.groups(k).t     = t(:)';
    dataset.groups(k).v_raw = v(:)';
    dataset.groups(k).u_raw = u(:)';

    all_v = [all_v, v(:)'];
    all_u = [all_u, u(:)'];
end

%% ====================== 5. 全局归一化 ==============================
fprintf("Computing global normalization...\n");

[all_v_norm, normIn]  = BasePINN.normalizeData(all_v, 'zscore');
[all_u_norm, normOut] = BasePINN.normalizeData(all_u, 'zscore');

cursor = 1;
for k = 1:numGroups
    N_k = numel(dataset.groups(k).v_raw);

    dataset.groups(k).v_norm = all_v_norm(cursor : cursor+N_k-1);
    dataset.groups(k).u_norm = all_u_norm(cursor : cursor+N_k-1);

    cursor = cursor + N_k;
end

dataset.normIn  = normIn;
dataset.normOut = normOut;
dataset.theta = theta;


%% ====================== 6. 保存 ==============================
save('dataset_GPI_dl.mat','dataset');
fprintf("Done. Saved as dataset_GPI_dl.mat\n");

%% (helper) 生成PRBS 序列
function v = prbs_signal(N)
    prbs_full = idinput(2047,'prbs');  % 完整序列
    v = prbs_full(1:N);
end
