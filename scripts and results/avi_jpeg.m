clc; clear all; close all;

filename = 'E:\image_processing\54ab110.avi';

mov=videoreader(filename); %�������e�̵ĵ�Ӱx.avi
%movie(mov); %��ӳ��Ӱ

%����Ӱת��ͼƬ����
fnum=size(mov,2); %��ȡ��Ӱ������
for i=1:fnum
    strtemp=strcat('C:\Users\tc\Desktop\test\',int2str(i),'.bmp');%��ÿ��ת��jpg��ͼƬ
    imwrite(mov(i).cdata,strtemp,'bmp');
end