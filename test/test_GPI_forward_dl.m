%% test_GPI_forward_dl.m
% 验证 GPI_forward_dl 与标量版 GPI_Model 的输出是否一致

clear; clc; close all;

%% 1. 正向 GPI 参数（与 GPI_Model 中保持一致）
theta.rd  = [-8 -6 -4 -2 0 2 4 6 8];
theta.eta = [1.0000 1.0000 1.0000 1.0000 1.2437 -0.1103 -0.0787 -0.1742 -0.1767];
theta.r   = [0 1 2 3 4 5];
theta.P   = [0.7460 0.1860 0.0650 0.0855 0.0362 -0.1159];

%% 2. 构造输入电压序列 v(t)
dt    = 1e-3;
T_end = 1.2;
t     = (0:dt:T_end-dt);           % 1 x N

% 多频正弦 + 限幅到 [0,10]，和 demo 里相同风格
v = 5 + 4*sin(2*pi*1*t) + 1.5*sin(2*pi*3*t);
v = max(0, min(10, v));

%% 3. 调用序列版 GPI_forward_dl
%  - 这里从“退磁状态”开始，Fr_init = []
[y_dl, Fr_final] = GPI_forward_dl(v, theta, []); 

% 将 dlarray 转为 double，方便后续对比
y_dl_num = double(extractdata(y_dl));

%% 4. 调用标量版 GPI_Model 做逐点递推
N       = numel(v);
y_ref   = zeros(1, N);
Fr_prev = [];

for k = 1:N
    [y_ref(k), Fr_prev] = GPI_Model(v(k), Fr_prev);
end

%% 5. 计算误差
err          = y_dl_num - y_ref;
max_abs_err  = max(abs(err));
rms_err      = sqrt(mean(err.^2));

fprintf('Max |error| = %.3e\n', max_abs_err);
fprintf('RMS error   = %.3e\n', rms_err);

%% 6. 画曲线对比
figure('Name', 'GPI_forward_dl vs GPI_Model'); 

subplot(2,1,1); hold on; box on;
plot(t, y_ref,    'k-',  'LineWidth', 1.4);
plot(t, y_dl_num, 'r--', 'LineWidth', 1.2);
xlabel('t [s]');
ylabel('y');
legend('GPI\_Model (标量版)', 'GPI\_forward\_dl (序列版)', 'Location', 'best');
title('GPI 正向模型输出对比');

subplot(2,1,2); hold on; box on;
plot(t, err, 'b-');
xlabel('t [s]');
ylabel('误差 y_{dl} - y_{ref}');
title(sprintf('误差曲线（Max |e| = %.3e, RMS = %.3e）', max_abs_err, rms_err));

grid on;
