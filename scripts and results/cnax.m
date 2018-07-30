% AX=AX';
% close all;
% clear all;
[m,n]=size(AX);
AX=AX';
x=AX(1,:);
for i=1:n-1
    y=AX(i+1,:);
      [y,PS]=mapminmax(y);
%      figure;
    plot(x,y);
    hold on;
   
%     axis(min(x),max(x),min(y(:,i)),max(y(:,i)));
end;


