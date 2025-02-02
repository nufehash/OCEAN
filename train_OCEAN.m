function [BB,XW,YW,HH] = train_OCEAN(XTrain_new,YTrain_new,LTrain_new,GTrain_new,BB,HH,param)
%
% This function is used to continue training the OCEAN algorithm, building upon the initial training done by train_OCEAN0. 
% It updates the hash codes and projection matrices for subsequent chunks of data.
%
% [BB,XW,YW,HH] = train_OCEAN(XTrain_new,YTrain_new,LTrain_new,GTrain_new,BB,HH,param);
  % Input   
  % XTrain_new: New chunk of image features for training.
  % YTrain_new: New chunk of text features for training.
  % LTrain_new: New chunk of labels for training.
  % GTrain_new: Normalized labels for training.
  % BB: Previously learned binary hash codes.
  % HH: Previously computed intermediate results.
  % param: A structure containing parameters.
  %
  % Output 
  % BB: Updated binary hash codes.
  % XW: Updated projection matrix for image features.
  % YW: Updated projection matrix for text features.
  % HH: Updated intermediate results.
%%
%%Reference:
% Online semantic embedding correlation for discrete cross-media hashing. 
% (Manuscript)
% Version1.0 -- Jan/2025
% Contant: Haoyu Hu (982258029@qq.com)
%%
%% Initialize parameters
    max_iter = param.max_iter;
    theta= param.theta;
    delta= param.delta;
    alpha = param.alpha; 
    beta = param.beta;
    mu= param.mu;
    nbits = param.nbits;
    n2 = size(LTrain_new,1);
    c = size(LTrain_new,2);
    max_iterf = 1;
    % Random initialization
    B_new = sign(randn(n2, nbits)); B_new(B_new==0) = -1;
    V_new = randn(n2, nbits);
    dX = size(XTrain_new,2);
    dY = size(YTrain_new,2);
    W=randn(c, c);
    G=(beta*(HH{1,15}+V_new'*V_new)+delta*eye(nbits))\(HH{1,5}+V_new'*(LTrain_new*W));
    % Iterative optimization
    for i = 1:max_iter
        %fprintf('iteration %3d\n', i);
        % Based on the optimization results from previous rounds (i.e., HH), 
        % optimize and update the relevant characters by incorporating data from newly arrived batches.
        P1=(theta*(HH{1,2}+XTrain_new'*XTrain_new)+delta*eye(dX))\(theta*(HH{1,1}+XTrain_new'*B_new));
        P2=(theta*(HH{1,17}+YTrain_new'*YTrain_new)+delta*eye(dY))\(theta*(HH{1,16}+YTrain_new'*B_new));
        Wtemp1=(beta+alpha)*(HH{1,4}+LTrain_new'*LTrain_new)+ delta*eye(c);
        Wtemp2=beta*(HH{1,3}+LTrain_new'*(V_new*G))+alpha*(HH{1,4}+LTrain_new'*LTrain_new);
        W=Wtemp1\Wtemp2;
        G=(beta*(HH{1,15}+V_new'*V_new)+delta*eye(nbits))\(HH{1,5}+V_new'*(LTrain_new*W));
        Z = B_new+beta*LTrain_new*W*G'...
            +mu*nbits*GTrain_new*(HH{1,6}+GTrain_new'*B_new);   
        Temp = Z'*Z-1/n2*(Z'*ones(n2,1)*(ones(1,n2)*Z));
        [~,Lmd,QQ] = svd(Temp); clear Temp
        idx = (diag(Lmd)>1e-6);     % 1e-6 is a set threshold below which singular values are considered negligible in their contribution to the data.
        Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
        P = (Z-1/n2*ones(n2,1)*(ones(1,n2)*Z)) *  (Q / (sqrt(Lmd(idx,idx))));
        P_ = orth(randn(n2,nbits-length(find(idx==1))));
        V_new = sqrt(n2)*[P P_]*[Q Q_]';
        B_new = sign(theta*XTrain_new*P1+theta*YTrain_new*P2+V_new...
                     +mu*nbits*GTrain_new*(HH{1,7}+GTrain_new'*V_new));
    end
    %% save results
    H1_new = XTrain_new'*B_new;
    H2_new = XTrain_new'*XTrain_new;
    H3_new = LTrain_new'*(V_new*G);
    H4_new = LTrain_new'*LTrain_new;
    H5_new = V_new'*(LTrain_new*W);
    H6_new = GTrain_new'*B_new;
    H7_new = GTrain_new'*V_new;
    % Building on optimization results from previous rounds, iterative updates are made by incorporating data from new rounds.
    HH{1,1} = HH{1,1}+H1_new;
    HH{1,2} = HH{1,2}+H2_new;
    HH{1,3} = HH{1,3}+H3_new;
    HH{1,4} = HH{1,4}+H4_new;
    HH{1,5} = HH{1,5}+H5_new;
    HH{1,6} = HH{1,6}+H6_new;
    HH{1,7} = HH{1,7}+H7_new;
    HH{1,15}  = HH{1,15}+ V_new'*V_new;
    HH{1,16}  = HH{1,16}+ YTrain_new'*B_new;
    HH{1,17}  = HH{1,17}+ YTrain_new'*YTrain_new;

%     BB{end+1,1} = B_new;
    
    
    % hash function learning
%     sel_num = floor(1000/param.nchunks);
%     if strcmp(param.db_name, 'NUS-WIDE')
%         sel_num = floor(5000/param.nchunks);
%     end
%     sel_idx = randperm(size(LTrain_new,1),sel_num);
% %      Bs_new = B_new(sel_idx,:);
%      Bs_new= B_new;
% %      Ss_new = GTrain_new*GTrain_new(sel_idx,:)';
%      Ss_new = GTrain_new*GTrain_new';
%     
%     HH{1,8}  =  HH{1,8}+Bs_new'*Bs_new;
%     HH{1,9}  =  HH{1,9}+XTrain_new'*Ss_new*Bs_new;
%     HH{1,10} =  HH{1,10}+YTrain_new'*Ss_new*Bs_new;
%     HH{1,11} =  HH{1,11}+XTrain_new'*B_new;
%     HH{1,12} =  HH{1,12}+YTrain_new'*B_new;
%     HH{1,13} =  HH{1,13}+XTrain_new'*XTrain_new;
%     HH{1,14} =  HH{1,14}+YTrain_new'*YTrain_new;
%     
%     
%     XW = (HH{1,13}+lambda*eye(dX))\(HH{1,11}+HH{1,9}*xi*nbits)...
%         /(eye(nbits)+HH{1,8}*xi);
%     YW = (HH{1,14}+lambda*eye(dY))\(HH{1,12}+HH{1,10}*xi*nbits)...
%         /(eye(nbits)+HH{1,8}*xi);
         M1 = zeros(size(B_new));
         M2 = M1;
         for iter = 1:max_iterf
             T1 = B_new + B_new.*M1;
             T2 = B_new + B_new.*M2;
             F1 = (HH{1,9}+XTrain_new'*XTrain_new+param.xi*eye(size(XTrain_new,2)))    \    (HH{1,11}+XTrain_new'*T1);
             F2 = (HH{1,10}+YTrain_new'*YTrain_new+param.xi*eye(size(YTrain_new,2)))    \    (HH{1,12}+YTrain_new'*T2);
             K1 = XTrain_new* F1-B_new;
             K2 = YTrain_new* F2-B_new;
             M1 = max(K1.*B_new,0);
             M2 = max(K2.*B_new,0);
         end
         XW=F1;
         YW=F2;
        %% save results
         H9_new = XTrain_new'*XTrain_new;
         H10_new = YTrain_new'*YTrain_new;
         H11_new = XTrain_new'*T1;
         H12_new = YTrain_new'*T2;
         HH{1,9} = HH{1,9}+H9_new;
         HH{1,10} = HH{1,10}+H10_new;
         HH{1,11} = HH{1,11}+H11_new;
         HH{1,12} = HH{1,12}+H12_new;
         
         BB{end+1,1} = B_new;


end
