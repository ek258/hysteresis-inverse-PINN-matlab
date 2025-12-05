function [y, Fr_final] = GPI_forward_dl(v, theta, Fr_init)
%GPI_FORWARD_DL  多阈值 GPI 正向模型的 dlarray 版本（整条序列）
%
%   [y, Fr_final] = GPI_forward_dl(v, theta, Fr_init)
%
%   v       : 1×N 或 N×1，电压序列，numeric 或 dlarray
%   theta   : 结构体，字段：
%               .rd   dead-zone 阈值，nD×1 或 1×nD  (对应原 rd)
%               .eta  dead-zone 权重，nD×1 或 1×nD  (对应原 eta)
%               .r    play 算子阈值，nR×1 或 1×nR   (对应原 r)
%               .P    输出权重，    nR×1 或 1×nR   (对应原 P)
%   Fr_init : 初始 play 状态，nR×1，可为空或省略，表示从“退磁”状态开始
%
%   y       : 1×N，位移输出（dlarray）
%   Fr_final: nR×1，最后时刻的 play 状态（numeric），可用于下一段的初值
%
%   对应你给的单步模型：
%       Ad(z) = dead-zone(v; rd(z))
%       w     = eta * Ad
%       Fr(j) = max( w - r(j), min(w + r(j), Fr_prev(j)) )
%       y     = P * Fr
%
%   注意：
%     - rd 目前在 dead-zone 分支里用 if rd(z)>0 / <0，因此更适合视作常量；
%       如需让 rd 也可学习，需要改成无分支的光滑近似（例如用 sign/tanh 逼近）。
%     - P, eta, r 完全支持作为可学习参数（dlarray）参与梯度。

    % ===== 0. v 转成 dlarray(single)，统一 shape：1×N =====
    if ~isa(v,'dlarray')
        v = dlarray(single(v));
    else
        if ~isa(extractdata(v),'single')
            v = dlarray(single(extractdata(v)));
        end
    end

    % 统一成 1×N 行向量
    if size(v,1) > 1 && size(v,2) == 1
        v = v.';
    end
    if size(v,1) ~= 1
        error('GPI_forward_dl:InputShape','v 必须是 1×N 或 N×1 向量');
    end

    N = size(v,2);

    % ===== 1. 取出参数并整理维度 =====
    % rd 仅用于比较和 dead-zone，不强制设为 dlarray，可视作常量
    rd = theta.rd(:).';    % 1×nD
    % 可学习参数：eta, r, P —— 全部用 dlarray(single)
    if isa(theta.eta,'dlarray')
        eta = theta.eta(:).';
    else
        eta = dlarray(single(theta.eta(:).'));
    end
    if isa(theta.r,'dlarray')
        r = theta.r(:);
    else
        r = dlarray(single(theta.r(:)));
    end
    if isa(theta.P,'dlarray')
        P = theta.P(:).';
    else
        P = dlarray(single(theta.P(:).'));
    end

    nD = numel(rd);
    nR = numel(r);

    % ===== 2. 初始 play 状态 =====
    if nargin < 3 || isempty(Fr_init)
        % 默认从“退磁状态”开始
        Fr_prev = dlarray(zeros(nR,1,'single'));
    else
        Fr_prev = Fr_init;
        if ~isa(Fr_prev,'dlarray')
            Fr_prev = dlarray(single(Fr_prev));
        end
        if ~isequal(size(Fr_prev), [nR,1])
            error('GPI_forward_dl:FrInitShape','Fr_init 尺寸应为 nR×1');
        end
    end

    % 输出序列
    y_data = dlarray(zeros(1,N,'single'));

    % ===== 3. 时间步递推 =====
    for k = 1:N
        vk = v(1,k);   % 当前电压

        % ----- 3.1 dead-zone 计算 Ad (nD×1) -----
        Ad = dlarray(zeros(nD,1,'single'));
        for z = 1:nD
            rz = rd(z);   % 这里 rd 作为常量使用
            if rz > 0
                Ad(z) = max(vk - rz, 0);
            elseif rz < 0
                Ad(z) = min(vk - rz, 0);
            else
                Ad(z) = vk;
            end
        end

        % w = eta * Ad
        % eta: 1×nD, Ad: nD×1 → scalar
        w = eta * Ad;

        % ----- 3.2 play 算子递推 Fr (nR×1) -----
        Fr = dlarray(zeros(nR,1,'single'));
        for j = 1:nR
            F_prev = Fr_prev(j);
            rj     = r(j);

            A = w - rj;
            B = min(w + rj, F_prev);
            Fr(j) = max(A, B);
        end

        % 更新内部状态
        Fr_prev = Fr;

        % ----- 3.3 输出 y(k) = P * Fr -----
        % P: 1×nR, Fr: nR×1 → scalar
        yk = P * Fr;
        y_data(1,k) = yk;
    end

    % ===== 4. 返回整条输出和最终状态 =====
    y = y_data;                    % 1×N, dlarray
    Fr_final = double(extractdata(Fr_prev));  % 保存成 numeric，方便跨批次用

end
