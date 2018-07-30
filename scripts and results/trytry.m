tic;
close all;
clear;
clc;
format compact;
%%


% ���ɴ��ع������
x = (-1:0.1:1)';
y = -x.^2;


% ��ģ�ع�ģ��
model = svmtrain(y,x,'-s 3 -t 2 -c 2.2 -g 2.8 -p 0.01');


% ���ý�����ģ�Ϳ�����ѵ�������ϵĻع�Ч��
[py,mse,devalue] = svmpredict(y,x,model);
figure;
plot(x,y,'o');
hold on;
plot(x,py,'r*');
legend('ԭʼ����','�ع�����');
grid on;


% ����Ԥ��
testx = [1.1,1.2,1.3]';
display('��ʵ����')
testy = -testx.^2


[ptesty,tmse,detesvalue] = svmpredict(testy,testx,model);
display('Ԥ������');
ptesty


%%
toc
