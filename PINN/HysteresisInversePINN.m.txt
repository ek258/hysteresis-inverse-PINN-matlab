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
    % X_data: 用于数据项和单调/平滑项的输入 (归一化后的 u_norm)
    % Y_data: 数据项目标输出 (归一化后的 v_norm)
    % X_phys: 用于物理残差 H(v)≈u 的输入 (归一化后的 u_norm_phys)
    %
    % 归一化:
    %   - 网络内部全部在归一化空间上工作 (u_norm, v_norm)
    %   - 物理项: 先把 u_norm, v_norm 反归一化为物理量 u_phys, v_phys
    %             再喂 physicsFcn(v_phys, paramsPhys)
    % -----------------------------------------------------------

    properties
        % 记录一份初始物理参数（原 physParam），只作为备查/重置用
        physParamInit      % struct，字段由外部决定，例如 {eta,r,P,...}

        lossWeights        % struct {lambdaData, lambdaPhys, lambdaMono, lambdaSmooth}
        physicsFcn         % 正向迟滞算子 (function handle)

        normIn             % 输入归一化参数 struct，字段至少包含 .mode
        normOut            % 输出归一化参数 struct，字段至少包含 .mode
    end

    methods
        %% ---------- 构造 ----------
        function obj = HysteresisInversePINN(layers, physParamInit, lossWeights, physicsFcn)
            % layers        : MLP 结构
            % physParamInit : 物理参数初值（numeric struct）
            %                 其中需要参与训练的字段，会被转成 dlarray(single)
            % lossWeights   : 各项损失权重
            % physicsFcn    : 物理前向算子句柄，例如
            %                 @(v,theta) GPI_forward_dl(v, theta_full, [])

            obj@BasePINN(layers);

            % ------- 物理参数初始化到 BasePINN.paramsPhys -------
            if nargin >= 2 && ~isempty(physParamInit)
                obj.physParamInit = physParamInit;

                fn = fieldnames(physParamInit);
                paramsPhys = struct();
                for i = 1:numel(fn)
                    name = fn{i};
                    val  = physParamInit.(name);

                    % 统一转成 dlarray(single) 以支持梯度
                    if isa(val, 'dlarray')
                        paramsPhys.(name) = val;
                    else
                        paramsPhys.(name) = dlarray(single(val));
                    end
                end
                obj.paramsPhys = paramsPhys;

                % 对物理参数初始化 Adam 状态
                [obj.adamAvgPhys, obj.adamAvgSqPhys] = obj.initAdamState(obj.paramsPhys);

                % 默认所有物理参数都可训练
                obj.trainablePhysParams = fieldnames(obj.paramsPhys);
            end

            if nargin >= 3 && ~isempty(lossWeights)
                obj.lossWeights = lossWeights;
            else
                obj.lossWeights = struct( ...
                    'lambdaData',   1.0, ...
                    'lambdaPhys',   0.0, ...
                    'lambdaMono',   0.0, ...
                    'lambdaSmooth', 0.0);
            end

            if nargin >= 4 && ~isempty(physicsFcn)
                obj.physicsFcn = physicsFcn;
            else
                obj.physicsFcn = [];
            end

            % 默认不做归一化
            obj.normIn  = struct('mode','none');
            obj.normOut = struct('mode','none');
        end

        %% 设置归一化参数（在 demo 中从训练数据统计后调用）
        function setNormalization(obj, normIn, normOut)
            if nargin >= 2 && ~isempty(normIn)
                obj.normIn = normIn;
            end
            if nargin >= 3 && ~isempty(normOut)
                obj.normOut = normOut;
            end
        end

        %% 物理域预测接口（方便 demo / 将来 S-function 使用）
        % u_phys: N×1 或 1×N (物理量)
        % v_phys: 同维度 (物理量)
        function v_phys = predictPhysical(obj, u_phys)
            u_row = u_phys(:)';   % 1 x N

            % 归一化输入
            u_norm = obj.applyNormIn(u_row);

            % 网络前向
            u_norm_dl = dlarray(single(u_norm));
            v_norm_dl = obj.forward(u_norm_dl);
            v_norm    = double(extractdata(v_norm_dl));

            % 反归一化输出
            v_row = obj.applyDenormOut(v_norm);
            v_phys = v_row(:);
        end

        %% ---------- 核心损失 ----------
        function [loss, gradsNet, gradsPhys] = computeLoss(obj, paramsNet, paramsPhys, ...
                                                           X_data, Y_data, X_phys, ...
                                                           physicsFcn, lossWeights)
            % X_data, Y_data, X_phys 均在归一化空间
            % X_*: 1 x N, dlarray(single)
            % Y_*: 1 x N, dlarray(single)

            if nargin < 7 || isempty(physicsFcn)
                physicsFcn = obj.physicsFcn;
            end
            if nargin < 8 || isempty(lossWeights)
                lossWeights = obj.lossWeights;
            end

            [L_data, L_phys, L_mono, L_smooth] = obj.computeLossTerms( ...
                paramsNet, paramsPhys, ...
                X_data, Y_data, X_phys, ...
                physicsFcn, lossWeights);

            % -------------------------------
            % 总损失
            % -------------------------------
            loss = ...
                lossWeights.lambdaData   * L_data   + ...
                lossWeights.lambdaPhys   * L_phys   + ...
                lossWeights.lambdaMono   * L_mono   + ...
                lossWeights.lambdaSmooth * L_smooth;

            % -------------------------------
            % 梯度：对网络参数 + 物理参数同时求导
            % -------------------------------
            [gradsNet, gradsPhys] = dlgradient(loss, paramsNet, paramsPhys);
        end

        function Lvec = evalLossComponents(obj, paramsNet, paramsPhys, ...
                                           X_data, Y_data, X_phys, ...
                                           physicsFcn, lossWeights)
            if nargin < 7 || isempty(physicsFcn)
                physicsFcn = obj.physicsFcn;
            end
            if nargin < 8 || isempty(lossWeights)
                lossWeights = obj.lossWeights;
            end

            [L_data, L_phys, L_mono, L_smooth] = obj.computeLossTerms( ...
                paramsNet, paramsPhys, ...
                X_data, Y_data, X_phys, ...
                physicsFcn, lossWeights);

            % ---------- 总损失 ----------
            L_total = ...
                lossWeights.lambdaData   * L_data + ...
                lossWeights.lambdaPhys   * L_phys + ...
                lossWeights.lambdaMono   * L_mono + ...
                lossWeights.lambdaSmooth * L_smooth;

            % 返回 vector（BasePINN 要求第一项必须是 total）
            Lvec = [L_total; L_data; L_phys; L_mono; L_smooth];
        end

    end

    methods (Access = private)
        function [L_data, L_phys, L_mono, L_smooth] = computeLossTerms(obj, paramsNet, paramsPhys, ...
                X_data, Y_data, X_phys, physicsFcn, lossWeights)
            % -------------------------------
            % 1) 数据前向：v_pred_data_norm = f(u_norm_data)
            % -------------------------------
            v_pred_data_norm = BasePINN.forwardWithParams(paramsNet, X_data);

            % -------------------------------
            % 2) 数据项 L_data (归一化空间)
            % -------------------------------
            if lossWeights.lambdaData ~= 0
                L_data = mean((v_pred_data_norm - Y_data).^2, "all");
            else
                L_data = dlarray(single(0));
            end

            % -------------------------------
            % 3) 物理一致性项 L_phys: H(v_phys_pred) ≈ u_phys
            %    先把 u_norm, v_norm 转回物理空间
            % -------------------------------
            if lossWeights.lambdaPhys ~= 0 && ~isempty(physicsFcn)
                % 3.1 归一化空间中预测
                v_pred_phys_norm = BasePINN.forwardWithParams(paramsNet, X_phys);

                % 3.2 反归一化到物理空间
                u_phys = obj.applyDenormIn(X_phys);            % u_phys
                v_phys = obj.applyDenormOut(v_pred_phys_norm); % v_phys

                % 3.3 调用正向迟滞模型得到 u_hat
                %     这里假设 physicsFcn 的签名为：
                %       u_hat = physicsFcn(v_phys, paramsPhys)
                u_hat = physicsFcn(v_phys, paramsPhys);

                % 3.4 在物理空间上计算残差
                L_phys = mean((u_hat - u_phys).^2, "all");
            else
                L_phys = dlarray(single(0));
            end

            % -------------------------------
            % 4) 单调性约束 L_mono （归一化空间 dv/du >= 0）
            % -------------------------------
            if lossWeights.lambdaMono ~= 0
                dv_du = dlgradient(sum(v_pred_data_norm, "all"), X_data);
                mono_penalty = relu(-dv_du);   % 只惩罚 dv/du<0 部分
                L_mono = mean(mono_penalty.^2, "all");
            else
                L_mono = dlarray(single(0));
            end

            % -------------------------------
            % 5) 输出平滑项 L_smooth （归一化空间）
            % -------------------------------
            if lossWeights.lambdaSmooth ~= 0
                v_seq = v_pred_data_norm;
                if size(v_seq,2) > 1
                    dv = v_seq(:,2:end) - v_seq(:,1:end-1);
                    L_smooth = mean(dv.^2, "all");
                else
                    L_smooth = dlarray(single(0));
                end
            else
                L_smooth = dlarray(single(0));
            end
        end

    end

    %% ========== 工具：dlarray 友好的归一化/反归一化 ==========
    methods (Access = public)
        % 对输入 u 做归一化 (网络输入前用)
        function u_n = applyNormIn(obj, u)
            % u: 1 x N (double or dlarray)
            mode = 'none';
            if isfield(obj.normIn, 'mode')
                mode = lower(obj.normIn.mode);
            end

            switch mode
                case 'zscore'
                    mu    = obj.normIn.mu;
                    sigma = obj.normIn.sigma;
                    if sigma == 0, sigma = 1; end
                    u_n = (u - mu) ./ sigma;

                case 'minmax'
                    xmin = obj.normIn.xmin;
                    xmax = obj.normIn.xmax;
                    denom = xmax - xmin;
                    if denom == 0, denom = 1; end
                    u_n = (u - xmin) ./ denom;

                otherwise
                    u_n = u;
            end
        end

        % 对输入的归一化值做反归一化 (物理项中恢复 u_phys)
        function u = applyDenormIn(obj, u_n)
            % u_n: 1 x N (dlarray)
            mode = 'none';
            if isfield(obj.normIn, 'mode')
                mode = lower(obj.normIn.mode);
            end

            switch mode
                case 'zscore'
                    mu    = obj.normIn.mu;
                    sigma = obj.normIn.sigma;
                    if sigma == 0, sigma = 1; end
                    u = u_n .* sigma + mu;

                case 'minmax'
                    xmin = obj.normIn.xmin;
                    xmax = obj.normIn.xmax;
                    denom = xmax - xmin;
                    if denom == 0, denom = 1; end
                    u = u_n .* denom + xmin;

                otherwise
                    u = u_n;
            end
        end

        % 对输出 v 的归一化值做反归一化 (物理项中恢复 v_phys)
        function v = applyDenormOut(obj, v_n)
            % v_n: 1 x N (dlarray 或 double)
            mode = 'none';
            if isfield(obj.normOut, 'mode')
                mode = lower(obj.normOut.mode);
            end

            switch mode
                case 'zscore'
                    mu    = obj.normOut.mu;
                    sigma = obj.normOut.sigma;
                    if sigma == 0, sigma = 1; end
                    v = v_n .* sigma + mu;

                case 'minmax'
                    xmin = obj.normOut.xmin;
                    xmax = obj.normOut.xmax;
                    denom = xmax - xmin;
                    if denom == 0, denom = 1; end
                    v = v_n .* denom + xmin;

                otherwise
                    v = v_n;
            end
        end
    end
end
