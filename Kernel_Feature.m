function [XKTrain,XKTest]=Kernel_Feature(XTrain,XTest,Anchors)
%
% This function used to generate kernel features.
% [XKTrain,XKTest]=Kernel_Feature(XTrain,XTest,Anchors)锛�
  % Input  
  % XTrain: Training dataset, with size [nX, Xdim], where nX is the number of samples and Xdim is the feature dimension.
  % XTest: Testing dataset, with size [nXT, XTdim], where nXT is the number of samples and XTdim is the feature dimension.
  % Anchors: Anchor points dataset, used to compute kernel features.
  % Output 
  % XKTrain: Kernel features of the training dataset.
  % XKTest: Kernel features of the testing dataset.
%
% % Reference:
% Online semantic embedding correlation for discrete cross-media hashing. 
% (Manuscript)
% Version1.0 -- Jan/2025
%
%
% Get the size of the datasets
% nX and Xdim are the number of samples and feature dimension of the training dataset, respectively.
% nXT and XTdim are the number of samples and feature dimension of the testing dataset, respectively.
    [nX,Xdim]=size(XTrain);
    [nXT,XTdim]=size(XTest);
%% Compute the kernel features of the training dataset
    XKTrain = sqdist(XTrain',Anchors');
    Xsigma = mean(mean(XKTrain,2));
    XKTrain = exp(-XKTrain/(2*Xsigma));
%     Xmvec = mean(XKTrain);
%     XKTrain = XKTrain-repmat(Xmvec,nX,1);
%% Compute the kernel features of the testing dataset 
    XKTest = sqdist(XTest',Anchors');
    XKTest = exp(-XKTest/(2*Xsigma));
%     XKTest = XKTest-repmat(Xmvec,nXT,1);
end
