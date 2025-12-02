classdef BasePINN < handle
    %BASEPINN Generic PINN base class.
    %   - 参数初始化
    %   - 前向传播 (MLP)
    %   - Adam 训练（可热启动）
    %   - 归一化工具
    %
    % 必须由子类实现：
    %   [loss, grads] = computeLoss(obj, params, X_data, Y_data, X_phys, physicsFcn, physParam, lossWeights)

    properties
        params          % 网络参数 W1,b1,...,WL,bL
        layers          % 网络结构
        lossHistory     % 损失曲线
        iteration       % Adam 迭代计数
        adamAvg
        adamAvgSq

        % 归一化设置
        inputNorm   % struct: mode='none'/'zscore'/'minmax'
        outputNorm
    end

    % ========================= 构造 + 接口方法 =========================
    methods
        %% 构造函数
        function obj = BasePINN(layers)
            obj.layers = layers;
            obj.params = obj.initParams(layers);
            [obj.adamAvg, obj.adamAvgSq] = obj.initAdamState(obj.params);
            obj.lossHistory = struct( ...
                'iter',      [], ...   % 1 x K
                'total',     [], ...   % 1 x K
                'components', [] );    % n_terms x K
            obj.iteration = 0;

            % 默认不开启归一化
            obj.inputNorm  = struct('mode','none');
            obj.outputNorm = struct('mode','none');
        end

        %% forward (使用当前参数)
        function Y = forward(obj, X)
            Y = BasePINN.forwardWithParams(obj.params, X);
        end

        %% train：使用 Adam
        function obj = train(obj, X_data, Y_data, X_phys, physicsFcn, physParam, lossWeights, epochs, lr)

            for ep = 1:epochs
                obj.iteration = obj.iteration + 1;

                % dlfeval 调用，注意参数顺序必须符合 lossWrapper 定义
                [loss, grads] = dlfeval(@BasePINN.lossWrapper, ...
                    obj, obj.params, ...
                    X_data, Y_data, X_phys, ...
                    physicsFcn, physParam, lossWeights);

                % Adam 更新
                fn = fieldnames(obj.params);
                for i = 1:numel(fn)
                    name = fn{i};

                    [obj.params.(name), obj.adamAvg.(name), obj.adamAvgSq.(name)] = ...
                        adamupdate( ...
                            obj.params.(name), grads.(name), ...
                            obj.adamAvg.(name), obj.adamAvgSq.(name), ...
                            obj.iteration, lr );
                end

                obj.lossHistory(end+1) = extractdata(loss);

                if mod(ep, 100) == 0
                    fprintf('Epoch %d / %d, Loss = %.3e\n', ep, epochs, extractdata(loss));
                end
            end
        end

        %% 保存模型
        function saveModel(obj, filename)
            save(filename, 'obj');
        end

        %% 抽象损失（必须由子类实现）
        function [loss, grads] = computeLoss(obj, params, ...
                                             X_data, Y_data, X_phys, ...
                                             physicsFcn, physParam, lossWeights)
            
            error('computeLoss must be implemented in subclass.');
        end

        % ========================= 归一化工具 =========================
        function setNormalization(obj, inputNorm, outputNorm)
            if nargin >= 2 && ~isempty(inputNorm)
                obj.inputNorm = inputNorm;
            end
            if nargin >= 3 && ~isempty(outputNorm)
                obj.outputNorm = outputNorm;
            end
        end

        function setNormalizationFromData(obj, X_data, Y_data, mode)
            if nargin < 4
                mode = 'zscore';
            end

            switch lower(mode)
                case 'zscore'
                    in_mean  = mean(X_data, 2, 'omitnan');
                    in_std   = std(X_data, 0, 2, 'omitnan');
                    in_std(in_std == 0) = 1;

                    out_mean = mean(Y_data, 2, 'omitnan');
                    out_std  = std(Y_data, 0, 2, 'omitnan');
                    out_std(out_std == 0) = 1;

                    obj.inputNorm = struct('mode','zscore', 'mean',in_mean, 'std',in_std);
                    obj.outputNorm = struct('mode','zscore', 'mean',out_mean, 'std',out_std);

                case 'minmax'
                    in_min = min(X_data,[],2);
                    in_max = max(X_data,[],2);
                    range_in = in_max - in_min;
                    range_in(range_in == 0) = 1;

                    out_min = min(Y_data,[],2);
                    out_max = max(Y_data,[],2);
                    range_out = out_max - out_min;
                    range_out(range_out == 0) = 1;

                    obj.inputNorm = struct('mode','minmax','min',in_min,'max',in_max,'range',range_in);
                    obj.outputNorm = struct('mode','minmax','min',out_min,'max',out_max,'range',range_out);

                otherwise
                    error('Unsupported normalization mode.');
            end
        end

        function Xn = normalizeInput(obj, X)
            Xn = BasePINN.applyNorm(X, obj.inputNorm, +1);
        end

        function X = denormalizeInput(obj, Xn)
            X = BasePINN.applyNorm(Xn, obj.inputNorm, -1);
        end

        function Yn = normalizeOutput(obj, Y)
            Yn = BasePINN.applyNorm(Y, obj.outputNorm, +1);
        end

        function Y = denormalizeOutput(obj, Yn)
            Y = BasePINN.applyNorm(Yn, obj.outputNorm, -1);
        end
    end

    % ========================= protected: 基类内部调用 =========================
    methods (Access = protected)
        function [loss, grads] = pinnLossInternal(obj, params, ...
                                                 X_data, Y_data, X_phys, ...
                                                 physicsFcn, physParam, lossWeights)
            [loss, grads] = obj.computeLoss(params, ...
                                            X_data, Y_data, X_phys, ...
                                            physicsFcn, physParam, lossWeights);
        end

        function params = initParams(~, layers)
            num_layers = numel(layers) - 1;
            params = struct();

            for i = 1:num_layers
                in_dim = layers(i);
                out_dim = layers(i+1);

                limit = sqrt(6/(in_dim + out_dim));   % Xavier 初始化

                W = rand(out_dim, in_dim)*2*limit - limit;
                b = zeros(out_dim, 1);

                params.(sprintf('W%d',i)) = dlarray(single(W));
                params.(sprintf('b%d',i)) = dlarray(single(b));
            end
        end

        function [stateAvg, stateAvgSq] = initAdamState(~, params)
            stateAvg   = struct();
            stateAvgSq = struct();

            fn = fieldnames(params);
            for i = 1:numel(fn)
                name = fn{i};

                p = params.(name);
                zero_like = dlarray(zeros(size(p), 'like', p));

                stateAvg.(name)   = zero_like;
                stateAvgSq.(name) = zero_like;
            end
        end
    end

    % ========================= 静态工具方法 =========================
    methods (Static)
        %% dlfeval 入口：必须确保参数顺序与 dlfeval 调用一致
        function [loss, grads] = lossWrapper(obj, params, ...
                                             X_data, Y_data, X_phys, ...
                                             physicsFcn, physParam, lossWeights)
            [loss, grads] = obj.pinnLossInternal(obj, params, ...
                    X_data, Y_data, X_phys, physicsFcn, physParam, lossWeights);
        end

        %% 前向传播（使用给定 params）
        function Y = forwardWithParams(params, X)
            if ~isa(X,'dlarray')
                X = dlarray(single(X));
            else
                X = dlarray(single(extractdata(X)));
            end

            % 自动统计层数
            num_layers = 0;
            while isfield(params, sprintf('W%d', num_layers+1))
                num_layers = num_layers + 1;
            end

            A = X;
            % 隐藏层 tanh
            for i = 1:(num_layers-1)
                W = params.(sprintf('W%d',i));
                b = params.(sprintf('b%d',i));

                Z = W*A + b;
                A = tanh(Z);
            end

            % 最后一层线性
            W = params.(sprintf('W%d',num_layers));
            b = params.(sprintf('b%d',num_layers));
            Y = W*A + b;
        end

        %% 归一化/反归一化通用函数
        function Z_out = applyNorm(Z_in, normStruct, direction)
            if nargin < 3
                direction = +1;
            end

            Z_out = Z_in;
            if isempty(normStruct) || ~isfield(normStruct,'mode')
                return;
            end
            if strcmp(normStruct.mode,'none')
                return;
            end

            switch normStruct.mode
                case 'zscore'
                    mu  = normStruct.mean;
                    sd  = normStruct.std;

                    if direction > 0
                        Z_out = (Z_in - mu) ./ sd;
                    else
                        Z_out = Z_in .* sd + mu;
                    end

                case 'minmax'
                    vmin  = normStruct.min;
                    vrange = normStruct.range;

                    if direction > 0
                        Z_out = (Z_in - vmin) ./ vrange;
                    else
                        Z_out = Z_in .* vrange + vmin;
                    end

                otherwise
                    error('Unsupported normalization mode.');
            end
        end

        %% 残差 + 随机混合采样
        function idx = residualRandomSample(residual, N_sample, fracRandom)
            if nargin < 3
                fracRandom = 0.5;
            end
            residual = residual(:);
            N_total = numel(residual);
            if N_total == 0
                idx = [];
                return;
            end

            N_sample = min(N_sample, N_total);
            N_rand = round(N_sample * fracRandom);
            N_top  = N_sample - N_rand;

            allIdx = (1:N_total).';

            % top K
            idx = [];
            if N_top > 0
                [~, I] = maxk(abs(residual), N_top);
                idx = I(:);
            end

            % random
            if N_rand > 0
                mask = true(N_total,1);
                mask(idx) = false;
                pool = allIdx(mask);

                if ~isempty(pool)
                    N_rand = min(N_rand, numel(pool));
                    rp = randperm(numel(pool), N_rand);
                    idx_rand = pool(rp);
                    idx = [idx; idx_rand(:)];
                end
            end

            idx = unique(idx, 'stable');
        end
    end
end
