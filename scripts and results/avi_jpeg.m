clc; clear all; close all;

filename = 'E:\image_processing\54ab110.avi';

mov=videoreader(filename); %读入存在e盘的电影x.avi
%movie(mov); %放映电影

%将电影转成图片序列
fnum=size(mov,2); %读取电影的祯数
for i=1:fnum
    strtemp=strcat('C:\Users\tc\Desktop\test\',int2str(i),'.bmp');%将每祯转成jpg的图片
    imwrite(mov(i).cdata,strtemp,'bmp');
end