data_all_ori=dataall;
for i=1:16
    data_all_ori(:,12*(17-i))=[];
end
for i=1:176
    dataimean=mean(data_all_ori(:,i));
    dataistd=std(data_all_ori(:,i));
    data_all_ori(:,i)=data_all_ori(:,i)-dataimean*ones(3000,1);
    data_all_ori(:,i)=data_all_ori(:,i)/dataistd;
    [data_all_orii,pr]=mapminmax(data_all_ori(:,i)');
    data_all_ori(:,i)=data_all_orii';
end
% ran1=randperm(750);ran2=randperm(750);ran3=randperm(750);ran4=randperm(750);
% data_all=[data_all_ori(ran1,:);data_all_ori(ran2+750,:);data_all_ori(ran3+1500,:);data_all_ori(ran4+2250,:)];

data_all=data_all_ori;
num_all=750;
% num_train_mat=[100 200 300 400 500 600];
% for j=1:6
% j=1;
% num_train=num_train_mat(j);

num_train=200;
num_test=750-num_train;
data_train=[data_all(1:num_train,:);data_all((1+num_all):(num_train+num_all),:);data_all((1+2*num_all):(num_train+2*num_all),:);data_all((1+3*num_all):(num_train+3*num_all),:)]';
data_test=[data_all((1+num_train):num_all,:);data_all((1+num_train+num_all):(2*num_all),:);data_all((1+num_train+2*num_all):(3*num_all),:);data_all((1+num_train+3*num_all):(4*num_all),:)]';

% label_all=[ones(num_all,1),zeros(num_all,1),zeros(num_all,1),zeros(num_all,1);
%     zeros(num_all,1),ones(num_all,1),zeros(num_all,1),zeros(num_all,1);
%     zeros(num_all,1),zeros(num_all,1),ones(num_all,1),zeros(num_all,1);
%     zeros(num_all,1),zeros(num_all,1),zeros(num_all,1),ones(num_all,1);]';
label_train=[ones(num_train,1),zeros(num_train,1),zeros(num_train,1),zeros(num_train,1);
    zeros(num_train,1),ones(num_train,1),zeros(num_train,1),zeros(num_train,1);
    zeros(num_train,1),zeros(num_train,1),ones(num_train,1),zeros(num_train,1);
    zeros(num_train,1),zeros(num_train,1),zeros(num_train,1),ones(num_train,1);]';
label_test=[ones(num_test,1),zeros(num_test,1),zeros(num_test,1),zeros(num_test,1);
    zeros(num_test,1),ones(num_test,1),zeros(num_test,1),zeros(num_test,1);
    zeros(num_test,1),zeros(num_test,1),ones(num_test,1),zeros(num_test,1);
    zeros(num_test,1),zeros(num_test,1),zeros(num_test,1),ones(num_test,1);]';
% label_train_svm=[zeros(1,num_train),ones(1,num_train),2.*ones(1,num_train),3.*ones(1,num_train)];
label_test_svm=[zeros(1,num_test),ones(1,num_test),2.*ones(1,num_test),3.*ones(1,num_test)];
% for i=1:10
% i=1;
net=patternnet(2);
    net.trainFcn='trainlm';
    [net,tr]=train(net,data_train,label_train);
%     serial0=1:11;

%     for j=1:10
%     for i=1:10000
%     ran_pot=randperm(16);
%     ran(i,(0.5*j*(j+1)+1):(0.5*j*(j+1)+j))=ran_pot(1:j);
% 
% data_test0=data_test;
% data_test0((11*(ran_pot(1:j)-1)+1):(11*ran_pot(1:j)),:)=0;
%     label_net=sim(net,data_test0);
%     [m ,index] = max(label_net,[],1) ;
%     num_accu=size(find(index-label_test_svm-1));
%     accu1(i)=100*(1-num_accu(1,2)/(4*num_test));
%     end;
%     [minaccu,index]=min(accu1);ran(index,:);
%     dot(j,1:10)=[index,zeros(1,10-j)];
%     end;

% for i=1:16
%     
%         data_test0=data_test;
%     for j=1:11
% 
% data_test0((16*(j-1)+i),:)=0;
%     end
%     label_net=sim(net,data_test0);
%     [m ,index] = max(label_net,[],1) ;
%     num_accu=size(find(index-label_test_svm-1));
%     accu5(i)=100*(1-num_accu(1,2)/(4*num_test));
%     
% end

% for i=1:11
%     data_train0=zeros(i*16,4*num_train);
%      data_test0=zeros(i*16,4*num_test);
% for j=1:16
% for m=1:i
% data_train0(16*(m-1)+j,:)=data_train((11*(j-1)+accu(2,m)),:);
% data_test0(16*(m-1)+j,:)=data_test((11*(j-1)+accu(2,m)),:);
% end
% end
% net=patternnet(2);
%     net.trainFcn='trainlm';
%     [net,tr]=train(net,data_train0,label_train);
%     label_net=sim(net,data_test0);
%     [m ,index] = max(label_net,[],1) ;
%     num_accu=size(find(index-label_test_svm-1));
%     accu1(3,i)=100*(1-num_accu(1,2)/(4*num_test));
% end

for i=1:16
    data_train0=zeros(i*16,4*num_train);
     data_test0=zeros(i*16,4*num_test);
for j=1:11

data_train0(j,:)=data_train((16*(j-1)+i),:);
data_test0(j,:)=data_test((16*(j-1)+i),:);

end
net=patternnet(2);
    net.trainFcn='trainlm';
    [net,tr]=train(net,data_train0,label_train);
    label_net=sim(net,data_test0);
    [m ,index] = max(label_net,[],1) ;
    num_accu=size(find(index-label_test_svm-1));
    accu1(4,i)=100*(1-num_accu(1,2)/(4*num_test));
end

% %     net1=patternnet(i);
% %     net1.trainFcn='trainlm';
% %     [net1,tr]=train(net1,data_train,label_train);
% %     label_net=sim(net1,data_test);
% %     [m ,index] = max(label_net,[],1) ;
% %     num_accu=size(find(index-label_test_svm-1));
% %     accu1(j,i)=100*(1-num_accu(1,2)/(4*num_test));
% 
% %     net2=patternnet(i);
% %     net2.trainFcn='trainbr';
% %     [net2,tr]=train(net2,data_train,label_train);
% %     label_net=sim(net2,data_test);
% %     [m ,index] = max(label_net,[],1) ;
% %     num_accu=size(find(index-label_test_svm-1));
% %     accu2(j,i)=100*(1-num_accu(1,2)/(4*num_test));
% %     
% %       net3=patternnet(i);
% %     net3.trainFcn='trainscg';
% %     [net3,tr]=train(net3,data_train,label_train);
% %       label_net=sim(net3,data_test);
% %     [m ,index] = max(label_net,[],1) ;
% %     num_accu=size(find(index-label_test_svm-1));
% %     accu3(j,i)=100*(1-num_accu(1,2)/(4*num_test));
% % end
% % end
% 
% 
% % label_all_svm=[zeros(1,num_all),ones(1,num_all),2.*ones(1,num_all),3.*ones(1,num_all)]';
% % label_train_svm=[zeros(1,num_train),ones(1,num_train),2.*ones(1,num_train),3.*ones(1,num_train)]';
% % label_test_svm=[zeros(1,num_test),ones(1,num_test),2.*ones(1,num_test),3.*ones(1,num_test)]';
% % nn=nnsetup([176 50 4]);
% % nn.dropoutFraction  =0.5;
% % opts.numepochs=1;
% % opts.batchsize=8;
% % [nn,L]=nntrain(nn,data_train,label_train,opts);
% % [er,bad]=nntest(nn,data_test,label_test);
% 
% 
% % data_pca=[data_train(:,1:11);
% % data_train(:,12:22);
% % data_train(:,23:33);
% % data_train(:,34:44);
% % data_train(:,45:55);
% % data_train(:,56:66);
% % data_train(:,67:77);
% % data_train(:,78:88);
% % data_train(:,89:99);
% % data_train(:,100:110);
% % data_train(:,111:121);
% % data_train(:,122:132);
% % data_train(:,133:143);
% % data_train(:,144:154);
% % data_train(:,155:165);
% % data_train(:,166:176);];
% % [coeff,score,latent,tsquared,explained]=pca(data_pca,'VariableWeights','variance');
% % coefforth=inv(diag(std(data_pca)))*coeff;
% % for i=1:11
% % dataplot(1,i)=i;
% % for j=1:6
% % dataplot(j+1,i)=coefforth(j,i)*explained(j);
% % end
% % end
% % figure;
% % pareto(explained);xlabel('主成分分析');ylabel('各成分包含信息（%）'):title('主成分包含信息');
% % figure;
% % biplot(coefforth(:,1:2),'scores',score(:,1:2));
% % figure;
% % biplot(coefforth(:,1:3),'scores',score(:,1:3));
% % figure;
% % plot(dataplot(1,:),sum(dataplot(2:6,:)),'r');xlabel('11个测量量');ylabel('测量量包含信息');title('各个测量量包含信息');hold on;
% 
% % for i=1:20
% % mdl=fitcknn(data_train,label_train_svm,'NumNeighbors',i,'DistanceWeight','equal');
% % label_test_knn=predict(mdl,data_test);
% % j=length(find((label_test_svm-label_test_knn)));
% % accu(i,1)=1-j/(4*num_test);
% % end
% % for i=1:20
% % mdl=fitcknn(data_train,label_train_svm,'NumNeighbors',i,'DistanceWeight','inverse');
% % label_test_knn=predict(mdl,data_test);
% % j=length(find((label_test_svm-label_test_knn)));
% % accu(i,2)=1-j/(4*num_test);
% % end
% % for i=1:20
% % mdl=fitcknn(data_train,label_train_svm,'NumNeighbors',i,'DistanceWeight','squaredinverse');
% % label_test_knn=predict(mdl,data_test);
% % j=length(find((label_test_svm-label_test_knn)));
% % accu(i,3)=1-j/(4*num_test);
% % end
% % 
% % plot(1:20,accu(:,1),'r',1:20,accu(:,2),'y',1:20,accu(:,3),'b');
% % title(['训练集',num2str(num_train),'测试集',num2str(num_test),'的KNN识别结果']);
% % xlabel('最近邻个数');
% % ylabel('识别准确率');
% % legend('距离等权重','倒数权重','倒数平方权重','Location','SouthEast')
% 
% 
% % serial=[2 3 4 5 6 8 10 ];
% % for j=1:10
% % model1 = svmtrain(label_all_svm, data_all, '-t 1 -d 2 -v 2');
% % model2 = svmtrain(label_all_svm, data_all, '-t 1 -d 3 -v 2');
% % model3 = svmtrain(label_all_svm, data_all, '-t 1 -d 4 -v 2');
% % model4 = svmtrain(label_all_svm, data_all, '-t 2 -g 0.5/176 -v 2');
% % model5 = svmtrain(label_all_svm, data_all, '-t 2 -g 1/176 -v 2');
% % model6 = svmtrain(label_all_svm, data_all, '-t 2 -g 2/176 -v 2');
% % model7 = svmtrain(label_all_svm, data_all, '-t 3 -v 2');
% % accu1(:,j)=[model1;model2;model3;model4;model5;model6;model7];
% % end
% % for j=1:10
% % 
% % model1 = svmtrain(label_all_svm, data_all, '-t 1 -d 2 -v 3');
% % model2 = svmtrain(label_all_svm, data_all, '-t 1 -d 3 -v 3');
% % model3 = svmtrain(label_all_svm, data_all, '-t 1 -d 4 -v 3');
% % model4 = svmtrain(label_all_svm, data_all, '-t 2 -g 0.5/176 -v 3');
% % model5 = svmtrain(label_all_svm, data_all, '-t 2 -g 1/176 -v 3');
% % model6 = svmtrain(label_all_svm, data_all, '-t 2 -g 2/176 -v 3');
% % model7 = svmtrain(label_all_svm, data_all, '-t 3 -v 3');
% % accu2(:,j)=[model1;model2;model3;model4;model5;model6;model7];
% % 
% % end
% % for j=1:10
% % model1 = svmtrain(label_all_svm, data_all, '-t 1 -d 2 -v 4');
% % model2 = svmtrain(label_all_svm, data_all, '-t 1 -d 3 -v 4');
% % model3 = svmtrain(label_all_svm, data_all, '-t 1 -d 4 -v 4');
% % model4 = svmtrain(label_all_svm, data_all, '-t 2 -g 0.5/176 -v 4');
% % model5 = svmtrain(label_all_svm, data_all, '-t 2 -g 1/176 -v 4');
% % model6 = svmtrain(label_all_svm, data_all, '-t 2 -g 2/176 -v 4');
% % model7 = svmtrain(label_all_svm, data_all, '-t 3 -v 2');
% % accu3(:,j)=[model1;model2;model3;model4;model5;model6;model7];
% % end
% % for j=1:10
% % 
% % model1 = svmtrain(label_all_svm, data_all, '-t 1 -d 2 -v 5');
% % model2 = svmtrain(label_all_svm, data_all, '-t 1 -d 3 -v 5');
% % model3 = svmtrain(label_all_svm, data_all, '-t 1 -d 4 -v 5');
% % model4 = svmtrain(label_all_svm, data_all, '-t 2 -g 0.5/176 -v 5');
% % model5 = svmtrain(label_all_svm, data_all, '-t 2 -g 1/176 -v 5');
% % model6 = svmtrain(label_all_svm, data_all, '-t 2 -g 2/176 -v 5');
% % model7 = svmtrain(label_all_svm, data_all, '-t 3 -v 5');
% % accu4(:,j)=[model1;model2;model3;model4;model5;model6;model7];
% % end
% % for j=1:10
% % 
% % model1 = svmtrain(label_all_svm, data_all, '-t 1 -d 2 -v 6');
% % model2 = svmtrain(label_all_svm, data_all, '-t 1 -d 3 -v 6');
% % model3 = svmtrain(label_all_svm, data_all, '-t 1 -d 4 -v 6');
% % model4 = svmtrain(label_all_svm, data_all, '-t 2 -g 0.5/176 -v 6');
% % model5 = svmtrain(label_all_svm, data_all, '-t 2 -g 1/176 -v 6');
% % model6 = svmtrain(label_all_svm, data_all, '-t 2 -g 2/176 -v 6');
% % model7 = svmtrain(label_all_svm, data_all, '-t 3 -v 6');
% % accu5(:,j)=[model1;model2;model3;model4;model5;model6;model7];
% % 
% % end
% % for j=1:10
% % 
% % model1 = svmtrain(label_all_svm, data_all, '-t 1 -d 2 -v 8');
% % model2 = svmtrain(label_all_svm, data_all, '-t 1 -d 3 -v 8');
% % model3 = svmtrain(label_all_svm, data_all, '-t 1 -d 4 -v 8');
% % model4 = svmtrain(label_all_svm, data_all, '-t 2 -g 0.5/176 -v 8');
% % model5 = svmtrain(label_all_svm, data_all, '-t 2 -g 1/176 -v 8');
% % model6 = svmtrain(label_all_svm, data_all, '-t 2 -g 2/176 -v 8');
% % model7 = svmtrain(label_all_svm, data_all, '-t 3 -v 8');
% % accu6(:,j)=[model1;model2;model3;model4;model5;model6;model7];
% % 
% % end
% % for j=1:10
% % model1 = svmtrain(label_all_svm, data_all, '-t 1 -d 2 -v 10');
% % model2 = svmtrain(label_all_svm, data_all, '-t 1 -d 3 -v 10');
% % model3 = svmtrain(label_all_svm, data_all, '-t 1 -d 4 -v 10');
% % model4 = svmtrain(label_all_svm, data_all, '-t 2 -g 0.5/176 -v 10');
% % model5 = svmtrain(label_all_svm, data_all, '-t 2 -g 1/176 -v 10');
% % model6 = svmtrain(label_all_svm, data_all, '-t 2 -g 2/176 -v 10');
% % model7 = svmtrain(label_all_svm, data_all, '-t 3 -v 10');
% % accu7(:,j)=[model1;model2;model3;model4;model5;model6;model7];