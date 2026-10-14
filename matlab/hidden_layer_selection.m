% 加载数据
filename = "D:\Desktop\四向量.xlsx";
data = readtable(filename, 'ReadVariableNames', false);

% 提取输入输出
input_data = data{:, 1:4};          % 前4列为维度隶属度
class_labels = categorical(data{:,5}); % 第5列为评语类别

% 检查类别数量
unique_labels = unique(class_labels);
num_classes = length(unique_labels);
disp(['识别到 ', num2str(num_classes), ' 个评语类别']);

% 将标签转换为one-hot编码
output_onehot = full(ind2vec(grp2idx(class_labels)', num_classes)';

% 数据归一化（输入已经是隶属度，无需再归一化）
% 直接转置为神经网络需要的格式
input_data = input_data';
output_data = output_onehot';

% 划分训练集和测试集 (70%-30%)
rng(2023); % 固定随机种子
n = size(input_data, 2);
idx = randperm(n);
train_num = round(0.7 * n);

trainInput = input_data(:, idx(1:train_num));
trainOutput = output_data(:, idx(1:train_num));
testInput = input_data(:, idx(train_num+1:end));
testOutput = output_data(:, idx(train_num+1:end));

% 设置隐藏层节点范围
hiddenLayerSize = 1:10;  % 建议初始测试范围
accuracyResults = zeros(length(hiddenLayerSize), 1); % 存储测试集准确率

% 神经网络参数
trainFcn = 'trainscg';   % 弹性BP算法，适合分类任务
maxEpochs = 500;

% 进度条
h = waitbar(0, '正在优化隐藏层...');

for i = 1:length(hiddenLayerSize)
    % 创建网络
    net = patternnet(hiddenLayerSize(i), trainFcn);
    
    % 配置网络参数
    net.trainParam.epochs = maxEpochs;
    net.trainParam.showWindow = false;
    net.performFcn = 'crossentropy'; % 交叉熵损失
    
    % 训练网络
    [net, tr] = train(net, trainInput, trainOutput);
    
    % 预测测试集
    testPred = net(testInput);
    [~, predLabels] = max(testPred);
    [~, trueLabels] = max(testOutput);
    
    % 计算准确率
    accuracy = sum(predLabels == trueLabels) / length(trueLabels);
    accuracyResults(i) = accuracy;
    
    % 更新进度
    waitbar(i/length(hiddenLayerSize), h, ...
        sprintf('当前节点数：%d，准确率：%.2f%%',...
        hiddenLayerSize(i), accuracy*100));
end
close(h);

% 结果分析
[bestAcc, bestIdx] = max(accuracyResults);
bestSize = hiddenLayerSize(bestIdx);

% 可视化
figure;
plot(hiddenLayerSize, accuracyResults*100, 'b-o');
xlabel('隐藏层神经元数量');
ylabel('测试集准确率 (%)');
title(['最优节点数：', num2str(bestSize), ' (准确率 ',...
       num2str(round(bestAcc*100,1)), '%)']);
grid on;

% 保存最佳配置
save('best_network.mat', 'net', 'bestSize');