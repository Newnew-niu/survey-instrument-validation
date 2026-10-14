net = feedforwardnet(9);  % 创建4-9-1网络
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';
view(net)
% 保存图片（运行后右键图形选择"Export"）