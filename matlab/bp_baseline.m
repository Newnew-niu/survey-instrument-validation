clc;
clear;

% 加载数据
data = xlsread('D:\Desktop\四向量.xlsx'); % 请确保路径正确

% 提取输入数据（前四列）和输出数据（最后一列）
X = data(:, 1:4); % 输入特征（4个输入）
y = data(:, 5); % 输出数据（1个输出）

% 选择训练集和测试集划分方式
trainRatio = 0.8;
testRatio = 1 - trainRatio;

% 随机划分训练集和测试集
N = size(X, 1);
randIdx = randperm(N); % 随机打乱数据

trainIdx = randIdx(1:round(N * trainRatio)); % 选择前80%作为训练集
testIdx = randIdx(round(N * trainRatio) + 1:end); % 剩下的20%作为测试集

X_train = X(trainIdx, :);
y_train = y(trainIdx);
X_test = X(testIdx, :);
y_test = y(testIdx);

% 创建一个神经网络，隐藏层神经元个数为9
hiddenNodes = 9;
net = feedforwardnet(hiddenNodes);

% 训练神经网络
net = train(net, X_train', y_train');

% 使用训练好的网络进行预测
y_pred = net(X_test');
y_pred = y_pred'; % 转置为与y_test相同的格式

% 计算性能指标：均方误差（MSE）、平均绝对误差（MAE）和准确率
MSE = mean((y_test - y_pred).^2);
MAE = mean(abs(y_test - y_pred));
accuracy = sum(round(y_pred) == y_test) / length(y_test); % 适用于分类问题

% 输出评估结果
fprintf('测试集上的 MSE: %.4f\n', MSE);
fprintf('测试集上的 MAE: %.4f\n', MAE);
fprintf('测试集上的 准确率: %.4f\n', accuracy);

% 绘制真实值与预测值的散点图
figure;
scatter(y_test, y_pred, 'filled');
xlabel('真实值');
ylabel('预测值');
title('真实值与预测值的散点图');
grid on;
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--'); % 完美预测线
legend('预测值 vs 真实值', '完美预测线');

% 计算残差并绘制残差图
residuals = y_test - y_pred;
figure;
scatter(y_test, residuals, 'filled');
xlabel('真实值');
ylabel('残差 (真实值 - 预测值)');
title('真实值与残差的散点图');
grid on;
hold on;
plot([min(y_test), max(y_test)], [0, 0], 'r--'); % 理想的残差线
legend('残差', '理想残差线');

% 绘制真实值与预测值的拟合曲线图
figure;
plot(y_test, 'b-', 'LineWidth', 2); % 真实值
hold on;
plot(y_pred, 'r--', 'LineWidth', 2); % 红色虚线显示预测值
xlabel('样本编号');
ylabel('值');
title('真实值与预测值的拟合曲线');
legend('真实值', '预测值');
grid on;

% 绘制预测值与真实值的折线图
figure;
% 绘制真实值的折线
plot(1:length(y_test), y_test, 'b-', 'LineWidth', 2); % 蓝色实线表示真实值
hold on;
% 绘制预测值的折线
plot(1:length(y_test), y_pred, 'r--', 'LineWidth', 2); % 红色虚线表示预测值
xlabel('样本编号');
ylabel('值');
title('真实值与预测值的折线图');
legend('真实值', '预测值');
grid on;
