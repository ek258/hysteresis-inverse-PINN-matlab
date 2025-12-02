classdef HysteresisInversePINN < BasePINN
    % -----------------------------------------------------------
    % Network A: 逆迟滞 PINN (u -> v_pred)
    %
    % 损失包括四项（每项由 lambda 控制，为0则跳过计算）：
    %   1) 数据项        L_data
    %   2) 物理一致性项  L_phys
    %   3) 单调性约束    L_mono
    %   4) 输出平滑项    L_smooth
    %
    % X_data: 用于数据项和单调/平滑项的输入 u
    % X_phys: 用于物理残差 H(v)≈u 的输入 u（可为子采样点）
    % -----------------------------------------------------------

    properties
        physParam       % 正向迟滞模型参数
        lossWeights     % 结构体 {lambdaData, lambdaPhys, lambdaMono, lambdaSmooth}
        physicsFcn      % 正向迟滞算子 (function handle)
    end

    methods
        %% ---------- 构造 ----------
        function obj = HysteresisInversePINN(layers, physParam, lossWeights, physicsFcn)
            obj@BasePINN(layers);
            obj.physParam   = physParam;
            obj.lossWeights = lossWeights;
            obj.physicsFcn  = physicsFcn;
        end

        %% ---------- 核心损失 ----------
        function [loss, grads] = computeLoss(~, params, ...
                                             X_data, Y_data, X_phys, ...
                                             physicsFcn, physParam, lossWeights)

            % -------------------------------
            % 1) 数据前向：v_pred_data = f(u_data)
            % -------------------------------
            v_pred_data = BasePINN.forwardWithParams(params, X_data);

            % -------------------------------
            % 2) 数据项 L_data
            % -------------------------------
            if lossWeights.lambdaData ~= 0
                L_data = mean((v_pred_data - Y_data).^2, "all");
            else
                L_data = dlarray(single(0));
            end

            % -------------------------------
            % 3) 物理一致性项 L_phys: H(v_pred_phys) ≈ u_phys
            %    v_pred_phys 在 X_phys 上前向（可抽样）
            % -------------------------------
            if lossWeights.lambdaPhys ~= 0
                v_pred_phys = BasePINN.forwardWithParams(params, X_phys);
                u_hat = physicsFcn(v_pred_phys, physParam);
                L_phys = mean((u_hat - X_phys).^2, "all");
            else
                L_phys = dlarray(single(0));
            end

            % -------------------------------
            % 4) 单调性约束 L_mono （在 X_data 上 dv/du >= 0）
            % -------------------------------
            if lossWeights.lambdaMono ~= 0
                dv_du = dlgradient(sum(v_pred_data), X_data);
                mono_penalty = relu(-dv_du);      % 只惩罚 dv/du<0 部分
                L_mono = mean(mono_penalty.^2, "all");
            else
                L_mono = dlarray(single(0));
            end

            % -------------------------------
            % 5) 输出平滑项 L_smooth （在 X_data 上）
            % -------------------------------
            if lossWeights.lambdaSmooth ~= 0
                v_seq = v_pred_data;
                if size(v_seq,2) > 1
                    dv = v_seq(:,2:end) - v_seq(:,1:end-1);
                    L_smooth = mean(dv.^2, "all");
                else
                    L_smooth = dlarray(single(0));
                end
            else
                L_smooth = dlarray(single(0));
            end

            % -------------------------------
            % 6) 总损失
            % -------------------------------
            loss = ...
                lossWeights.lambdaData   * L_data   + ...
                lossWeights.lambdaPhys   * L_phys   + ...
                lossWeights.lambdaMono   * L_mono   + ...
                lossWeights.lambdaSmooth * L_smooth;

            % -------------------------------
            % 7) 梯度
            % -------------------------------
            grads = dlgradient(loss, params);
        end

        %% ---------- 拆分各项损失（给 Base / demo 记录用） ----------
        function Lvec = evalLossComponents(~, params, ...
                                           X_data, Y_data, X_phys, ...
                                           physicsFcn, physParam, lossWeights)
            % 与 computeLoss 类似，但返回数值向量:
            % Lvec = [Ltotal, Ldata, Lphys, Lmono, Lsmooth]

            % 1) data forward
            v_pred_data = BasePINN.forwardWithParams(params, X_data);

            % 2) L_data
            if lossWeights.lambdaData ~= 0
                L_data_dl = mean((v_pred_data - Y_data).^2, "all");
                Ldata = double(extractdata(L_data_dl));
            else
                Ldata = 0;
            end

            % 3) L_phys
            if lossWeights.lambdaPhys ~= 0
                v_pred_phys = BasePINN.forwardWithParams(params, X_phys);
                u_hat = physicsFcn(v_pred_phys, physParam);
                L_phys_dl = mean((u_hat - X_phys).^2, "all");
                Lphys = double(extractdata(L_phys_dl));
            else
                Lphys = 0;
            end

            % 4) L_mono
            if lossWeights.lambdaMono ~= 0
                dv_du = dlgradient(sum(v_pred_data), X_data);
                mono_penalty = relu(-dv_du);
                L_mono_dl = mean(mono_penalty.^2, "all");
                Lmono = double(extractdata(L_mono_dl));
            else
                Lmono = 0;
            end

            % 5) L_smooth
            if lossWeights.lambdaSmooth ~= 0 && size(v_pred_data,2) > 1
                dv = v_pred_data(:,2:end) - v_pred_data(:,1:end-1);
                L_smooth_dl = mean(dv.^2, "all");
                Lsmooth = double(extractdata(L_smooth_dl));
            else
                Lsmooth = 0;
            end

            % 6) 总损失（按 lambda 加权）
            Ltotal = ...
                lossWeights.lambdaData   * Ldata   + ...
                lossWeights.lambdaPhys   * Lphys   + ...
                lossWeights.lambdaMono   * Lmono   + ...
                lossWeights.lambdaSmooth * Lsmooth;

            Lvec = [Ltotal, Ldata, Lphys, Lmono, Lsmooth];
        end
    end
end
