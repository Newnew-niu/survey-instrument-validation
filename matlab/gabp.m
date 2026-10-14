% 1. 加载数据
data = xlsread('D:\Desktop\四向量.xlsx'); % 数据路径已更新

X = data(:, 1:4); % 输入特征（4个输入）
y = data(:, 5);   % 输出数据（1个输出）

% 2. 数据标准化
X = (X - min(X)) ./ (max(X) - min(X)); % 对输入数据进行归一化

% 3. 划分训练集和测试集
trainRatio = 0.8; % 80% 数据用于训练
testRatio = 1 - trainRatio; % 20% 数据用于测试
N = size(X, 1);

randIdx = randperm(N);
trainIdx = randIdx(1:round(N * trainRatio)); % 80% 训练集
testIdx = randIdx(round(N * trainRatio) + 1:end); % 20% 测试集

X_train = X(trainIdx, :);
y_train = y(trainIdx);
X_test = X(testIdx, :);
y_test = y(testIdx);

% 4. 设置遗传算法参数
populationSize = 30; % 种群大小
numGenerations = 30; % 最大代数
mutationRate = 0.1; % 变异率
crossoverRate = 0.7; % 交叉率

% 计算神经网络的权重数量
inputSize = size(X, 2); % 输入层神经元数目（4个）
hiddenNodes = 9; % 隐藏层神经元数目（9个） - 修正过来的地方
outputSize = 1; % 输出层神经元数目（1个）

% 总的权重数量：4 * 9 + 9 + 9 * 1 + 1
numWeights = inputSize * hiddenNodes + hiddenNodes + hiddenNodes * outputSize + outputSize;

% 5. 初始化遗传算法种群
population = rand(populationSize, numWeights) * 2 - 1; % 初始化种群

% 遗传算法优化
for generation = 1:numGenerations
    % 评估适应度
    fitness = evaluate_fitness(population, X_train, y_train, hiddenNodes);
    
    % 选择操作：锦标赛选择
    [~, idx] = sort(fitness, 'descend');
    population = population(idx, :); % 根据适应度排序

    % 交叉操作：单点交叉
    for i = 1:2:populationSize
        if rand < crossoverRate
            crossoverPoint = randi([1, numWeights-1]); % 随机选择交叉点
            temp = population(i, crossoverPoint+1:end);
            population(i, crossoverPoint+1:end) = population(i+1, crossoverPoint+1:end);
            population(i+1, crossoverPoint+1:end) = temp;
        end
    end
    
    % 变异操作：随机变异
    for i = 1:populationSize
        if rand < mutationRate
            mutationPoint = randi([1, numWeights]); % 随机选择变异点
            population(i, mutationPoint) = rand * 2 - 1; % 随机赋值
        end
    end
    
    % 输出遗传算法当前代数及适应度信息
    if mod(generation, 10) == 0
        fprintf('Generation %d, Best Fitness: %.4f\n', generation, max(fitness));
    end
end

% 6. 使用最优解训练BP神经网络
best_weights = population(1, :);
net = feedforwardnet(hiddenNodes);
net = configure(net, X_train', y_train');
net = set_weights(net, best_weights);

% 添加L2正则化
net.performParam.regularization = 0.01; % L2 正则化

% 设置早停法
net.trainParam.max_fail = 6; % 如果验证误差连续6次没有改善，则停止训练

% 训练网络
disp('Training the network...');
[net, tr] = train(net, X_train', y_train');
disp('Training completed.');

% 7. 预测与评估
disp('Performing predictions...');
y_pred = net(X_test');
MSE_test = mean((y_test - y_pred').^2); % 计算测试集上的均方误差
fprintf('测试集上的均方误差：%.4f\n', MSE_test);

% 8. 输出具体的预测结果和实际值对比
disp('实际值与预测值对比：');
result_table = table(X_test, y_test, y_pred', 'VariableNames', {'输入特征', '实际值', '预测值'});
disp(result_table);

% 9. 绘制真实值与预测值的对比
figure;
plot(y_test, 'b-', 'LineWidth', 2); % 真实值
hold on;
plot(y_pred, 'r--', 'LineWidth', 2); % 预测值
xlabel('样本编号');
ylabel('值');
title('真实值与预测值的拟合曲线');
legend('真实值', '预测值');
grid on;

% 以下是函数部分，放在脚本末尾

% 适应度函数：计算每个个体的MSE作为适应度
function fitness = evaluate_fitness(population, X_train, y_train, hiddenNodes)
    fitness = zeros(size(population, 1), 1); % 初始化适应度数组
    for i = 1:size(population, 1)
        % 创建神经网络
        net = feedforwardnet(hiddenNodes);
        net = configure(net, X_train', y_train'); % 配置网络结构
        net = set_weights(net, population(i, :)); % 设置优化后的权重
        y_pred = net(X_train');
        MSE = mean((y_train - y_pred').^2); % 计算MSE
        fitness(i) = 1 / (1 + MSE); % 适应度 = MSE的倒数
    end
end

% 设置神经网络的权重
function net = set_weights(net, weights)
    % 输入到隐藏层的权重（9×4）
    net.IW{1,1} = reshape(weights(1:36), 9, 4); 
    % 隐藏层的偏置（9个）
    net.b{1} = reshape(weights(37:45), 9, 1); 
    % 隐藏层到输出层的权重（1×9）
    net.LW{2,1} = reshape(weights(46:54), 1, 9); 
    % 输出层的偏置（1个）
    net.b{2} = weights(55); 
end
