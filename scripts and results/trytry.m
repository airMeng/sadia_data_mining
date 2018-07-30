tic;
close all;
clear;
clc;
format compact;
%%


% 生成待回归的数据
x = (-1:0.1:1)';
y = -x.^2;


% 建模回归模型
model = svmtrain(y,x,'-s 3 -t 2 -c 2.2 -g 2.8 -p 0.01');


% 利用建立的模型看其在训练集合上的回归效果
[py,mse,devalue] = svmpredict(y,x,model);
figure;
plot(x,y,'o');
hold on;
plot(x,py,'r*');
legend('原始数据','回归数据');
grid on;


% 进行预测
testx = [1.1,1.2,1.3]';
display('真实数据')
testy = -testx.^2


[ptesty,tmse,detesvalue] = svmpredict(testy,testx,model);
display('预测数据');
ptesty


%%
toc
