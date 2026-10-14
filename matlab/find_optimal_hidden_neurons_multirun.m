% 读取数据
data = xlsread('D:\Desktop\四向量.xlsx');
X = data(:, 1:4); % 输入特征
y = data(:, 5); % 输出标签

% 数据归一化
X = normalize(X);

% 交叉验证设置
k = 5; % 折数
cv = cvpartition(length(y), 'KFold', k);

% 尝试不同隐藏层神经元个数
hidden_neurons_range = 3:15;

num_runs = 10; % 运行次数
counts = zeros(size(hidden_neurons_range));

for run = 1:num_runs
    mse_results = zeros(size(hidden_neurons_range));
    
    for i = 1:length(hidden_neurons_range)
        hidden_neurons = hidden_neurons_range(i);
        mse_sum = 0;
        
        % 进行k折交叉验证
        for j = 1:k
            train_idx = training(cv, j);
            test_idx = test(cv, j);
            
            X_train = X(train_idx, :);
            y_train = y(train_idx);
            X_test = X(test_idx, :);
            y_test = y(test_idx);
            
            % 创建BP神经网络
            net = feedforwardnet(hidden_neurons);
            net.trainFcn = 'trainlm'; % 使用Levenberg - Marquardt算法训练
            net.performFcn = 'mse'; % 使用均方误差作为性能指标
            
            % 加入L2正则化
            net.trainParam.lambda = 0.01; 
            
            % 训练网络
            net = train(net, X_train', y_train');
            
            % 预测
            y_pred = net(X_test');
            
            % 计算均方误差
            mse = mean((y_pred - y_test').^2);
            mse_sum = mse_sum + mse;
        end
        
        % 计算平均均方误差
        mse_results(i) = mse_sum / k;
    end
    
    % 选择最优隐藏层神经元个数
    [~, best_idx] = min(mse_results);
    best_hidden_neurons = hidden_neurons_range(best_idx);
    
    % 统计出现次数
    idx = find(hidden_neurons_range == best_hidden_neurons);
    counts(idx) = counts(idx) + 1;
end

% 找出出现次数最多的隐藏层神经元个数
[~, final_best_idx] = max(counts);
final_best_hidden_neurons = hidden_neurons_range(final_best_idx);
disp(['最终最优隐藏层神经元个数: ', num2str(final_best_hidden_neurons)]);

