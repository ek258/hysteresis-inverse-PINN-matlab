classdef BasePINN < handle
    %BASEPINN Generic PINN base class.
    %   - 管理参数初始化
    %   - 前向传播 (MLP)
    %   - Adam 训练 + 热启动
    %   - dlfeval 通过静态 wrapper 统一调用
    %
    % 子类必须实现:
    %   [loss, grads] = computeLoss(obj, params, X_data, Y_data, X_phys, physicsFcn, physParam, lossWeights)

    properties
        params          % W1,b1,...,WL,bL
        layers
        lossHistory     % 通用损失记录: struct('iter',[],'total',[],'components',[])
        iteration       % Adam 迭代计数（支持热启动）
        adamAvg
        adamAvgSq
    end

    methods
        %% 构造函数
        function obj = BasePINN(layers)
            obj.layers = layers;
            obj.params = obj.initParams(layers);
            [obj.adamAvg, obj.adamAvgSq] = obj.initAdamState(obj.params);

            % 通用的 loss 记录结构
            obj.lossHistory = struct( ...
                'iter',      [], ...   % 1 x K
                'total',     [], ...   % 1 x K
                'components', [] );    % n_terms x K

            obj.iteration = 0;
        end

        %% 前向预测，使用当前 obj.params
        function Y = forward(obj, X)
            Y = BasePINN.forwardWithParams(obj.params, X);
        end

        %% 训练（Adam），支持热启动
        function obj = train(obj, X_data, Y_data, X_phys, ...
                             physicsFcn, physParam, lossWeights, ...
                             epochs, lr)

            for ep = 1:epochs
                obj.iteration = obj.iteration + 1;

                % 通过静态 wrapper + dlfeval 调用 computeLoss
                [loss, grads] = dlfeval(@BasePINN.lossWrapper, ...
                    obj, obj.params, ...
                    X_data, Y_data, X_phys, ...
                    physicsFcn, physParam, lossWeights);

                % Adam 更新（逻辑和原 trainPINN 完全一致）
                fn = fieldnames(obj.params);
                for i = 1:numel(fn)
                    name = fn{i};

                    [obj.params.(name), ...
                     obj.adamAvg.(name), ...
                     obj.adamAvgSq.(name)] = ...
                        adamupdate( ...
                            obj.params.(name), grads.(name), ...
                            obj.adamAvg.(name), obj.adamAvgSq.(name), ...
                            obj.iteration, lr);
                end

                % ===== 通用记录损失（如果子类实现了 evalLossComponents）=====
                if ismethod(obj, 'evalLossComponents')
                    % 返回一个向量: [Ltotal, L1, L2, ...]
                    Lvec = dlfeval(@BasePINN.evalWrapper, ...
                                   obj, obj.params, ...
                                   X_data, Y_data, X_phys, ...
                                   physicsFcn, physParam, lossWeights);
                    Lvec = double(Lvec(:)).';  % 1 x n_terms

                    obj.lossHistory.iter(end+1)  = obj.iteration;
                    obj.lossHistory.total(end+1) = Lvec(1);

                    if isempty(obj.lossHistory.components)
                        % 第一次: 初始化为 n_terms x 1
                        obj.lossHistory.components = Lvec(:);
                    else
                        % 之后: 在第二维追加一列
                        if size(obj.lossHistory.components,1) ~= numel(Lvec)
                            error('Loss component size mismatch between iterations.');
                        end
                        obj.lossHistory.components(:, end+1) = Lvec(:);
                    end
                else
                    % 如果没实现 evalLossComponents，也至少记录总损失
                    Ltotal = double(extractdata(loss));
                    obj.lossHistory.iter(end+1)  = obj.iteration;
                    obj.lossHistory.total(end+1) = Ltotal;
                    % components 留空
                end

                if mod(ep, 100) == 0
                    fprintf('Epoch %d / %d, Loss = %.3e\n', ep, epochs, extractdata(loss));
                end
            end
        end

        %% 保存模型
        function saveModel(obj, filename)
            save(filename, 'obj');
        end

        %% 抽象损失（子类实现）
        function [loss, grads] = computeLoss(obj, params, ...
                                             X_data, Y_data, X_phys, ...
                                             physicsFcn, physParam, lossWeights)
            %#ok<INUSD>
            error('computeLoss must be implemented in subclass.');
        end
    end

    %% ========== 受保护的方法（基类内部/子类可用） ==========
    methods (Access = protected)
        % 调用子类 computeLoss 的内部接口
        function [loss, grads] = pinnLossInternal(obj, params, ...
                                                 X_data, Y_data, X_phys, ...
                                                 physicsFcn, physParam, lossWeights)
            [loss, grads] = obj.computeLoss(params, ...
                                            X_data, Y_data, X_phys, ...
                                            physicsFcn, physParam, lossWeights);
        end

        % 参数初始化（等价于原 initPINNParams）
        function params = initParams(~, layers)
            num_layers = numel(layers) - 1;
            params = struct();

            for i = 1:num_layers
                in_dim  = layers(i);
                out_dim = layers(i+1);

                limit = sqrt(6/(in_dim + out_dim));   % Xavier uniform

                W = rand(out_dim, in_dim)*2*limit - limit;
                b = zeros(out_dim, 1);

                params.(sprintf('W%d', i)) = dlarray(single(W));
                params.(sprintf('b%d', i)) = dlarray(single(b));
            end
        end

        % Adam 状态初始化（等价于原 initAdamState）
        function [stateAvg, stateAvgSq] = initAdamState(~, params)
            stateAvg   = struct();
            stateAvgSq = struct();

            fn = fieldnames(params);
            for i = 1:numel(fn)
                name = fn{i};

                p = params.(name);
                zeros_like = dlarray(zeros(size(p), 'like', p));

                stateAvg.(name)   = zeros_like;
                stateAvgSq.(name) = zeros_like;
            end
        end
    end

    %% ========== 静态方法（无状态工具函数） ==========
    methods (Static)
        % dlfeval 的统一入口（用于 computeLoss）
        function [loss, grads] = lossWrapper(obj, params, ...
                                             X_data, Y_data, X_phys, ...
                                             physicsFcn, physParam, lossWeights)
            [loss, grads] = obj.pinnLossInternal(params, ...
                                X_data, Y_data, X_phys, ...
                                physicsFcn, physParam, lossWeights);
        end

        % dlfeval 入口（用于 evalLossComponents）
        function Lvec = evalWrapper(obj, params, ...
                                    X_data, Y_data, X_phys, ...
                                    physicsFcn, physParam, lossWeights)
            Lvec = obj.evalLossComponents(params, ...
                                          X_data, Y_data, X_phys, ...
                                          physicsFcn, physParam, lossWeights);
        end

        % 前向传播（等价于原 forwardPINN），可给任意 params
        function Y = forwardWithParams(params, X)
            % 确保 X 是 dlarray(single)
            if ~isa(X, 'dlarray')
                X = dlarray(single(X));
            else
                if ~isa(extractdata(X), 'single')
                    X = dlarray(single(extractdata(X)));
                end
            end

            % 自动统计层数
            num_layers = 0;
            while isfield(params, sprintf('W%d', num_layers + 1))
                num_layers = num_layers + 1;
            end

            A = X;
            % 前 num_layers-1 层 tanh
            for i = 1:(num_layers-1)
                W = params.(sprintf('W%d', i));
                b = params.(sprintf('b%d', i));

                Z = W*A + b;
                A = tanh(Z);
            end

            % 最后一层线性
            W = params.(sprintf('W%d', num_layers));
            b = params.(sprintf('b%d', num_layers));

            Y = W*A + b;
        end

        % ========= 通用工具函数：归一化 / 反归一化 =========
        function [Xn, normInfo] = normalizeData(X, mode, normInfo)
            %NORMALIZEDATA  对列样本数据做归一化（特征在行）
            %
            %   [Xn, normInfo] = BasePINN.normalizeData(X, mode)
            %   [Xn, normInfo] = BasePINN.normalizeData(X, mode, normInfo)
            %
            %   mode: 'none' | 'zscore' | 'minmax'
            %   X   : (d x N) 矩阵，d 维特征，N 个样本
            %
            %   normInfo 结构:
            %     .mode  = mode
            %   对于 'zscore':
            %     .mu    : d x 1
            %     .sigma : d x 1
            %   对于 'minmax':
            %     .xmin  : d x 1
            %     .xmax  : d x 1

            if nargin < 2 || isempty(mode)
                mode = 'none';
            end
            if nargin < 3
                normInfo = struct();
            end

            Xn = X;
            modeLower = lower(mode);

            switch modeLower
                case 'none'
                    normInfo.mode = 'none';

                case 'zscore'
                    if ~isfield(normInfo, 'mu') || ~isfield(normInfo, 'sigma')
                        normInfo.mu    = mean(X, 2);
                        normInfo.sigma = std(X, 0, 2);
                        % 防止除 0
                        normInfo.sigma(normInfo.sigma == 0) = 1;
                    end
                    normInfo.mode = 'zscore';
                    Xn = (X - normInfo.mu) ./ normInfo.sigma;

                case 'minmax'
                    if ~isfield(normInfo, 'xmin') || ~isfield(normInfo, 'xmax')
                        normInfo.xmin = min(X, [], 2);
                        normInfo.xmax = max(X, [], 2);
                    end
                    normInfo.mode = 'minmax';

                    denom = normInfo.xmax - normInfo.xmin;
                    denom(denom == 0) = 1;

                    Xn = (X - normInfo.xmin) ./ denom;

                otherwise
                    error('BasePINN:normalizeData', ...
                          'Unknown normalization mode "%s".', mode);
            end
        end

        function X = denormalizeData(Xn, normInfo)
            %DENORMALIZEDATA  根据 normInfo 做反归一化
            %
            %   X = BasePINN.denormalizeData(Xn, normInfo)

            if ~isfield(normInfo, 'mode') || strcmpi(normInfo.mode, 'none')
                X = Xn;
                return;
            end

            modeLower = lower(normInfo.mode);

            switch modeLower
                case 'zscore'
                    mu    = normInfo.mu;
                    sigma = normInfo.sigma;
                    sigma(sigma == 0) = 1;
                    X = Xn .* sigma + mu;

                case 'minmax'
                    xmin = normInfo.xmin;
                    xmax = normInfo.xmax;
                    denom = xmax - xmin;
                    denom(denom == 0) = 1;
                    X = Xn .* denom + xmin;

                otherwise
                    error('BasePINN:denormalizeData', ...
                          'Unknown normalization mode "%s".', normInfo.mode);
            end
        end

        % ========= 通用工具函数：train / test 切分 =========
        function [Xtrain, Ytrain, Xtest, Ytest, idxTrain, idxTest] = ...
                 trainTestSplit(X, Y, testRatio, doShuffle)
            %TRAINTESTSPLIT  按列切分训练/测试集
            %
            %   [Xtr,Ytr,Xte,Yte,idxTr,idxTe] = BasePINN.trainTestSplit(X, Y, testRatio, doShuffle)
            %
            %   X: d_in x N
            %   Y: d_out x N
            %   testRatio: 测试集比例，默认 0.2
            %   doShuffle: 是否打乱，默认 true

            if nargin < 3 || isempty(testRatio)
                testRatio = 0.2;
            end
            if nargin < 4 || isempty(doShuffle)
                doShuffle = true;
            end

            N = size(X, 2);
            if ~isempty(Y) && size(Y, 2) ~= N
                error('BasePINN:trainTestSplit', ...
                      'X and Y must have the same number of columns (samples).');
            end

            nTest  = max(1, floor(N * testRatio));
            nTrain = N - nTest;

            if doShuffle
                idx = randperm(N);
            else
                idx = 1:N;
            end

            idxTrain = idx(1:nTrain);
            idxTest  = idx(nTrain+1:end);

            Xtrain = X(:, idxTrain);
            Xtest  = X(:, idxTest);

            if isempty(Y)
                Ytrain = [];
                Ytest  = [];
            else
                Ytrain = Y(:, idxTrain);
                Ytest  = Y(:, idxTest);
            end
        end
    end
end
