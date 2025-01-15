function [XChunk,YChunk,LChunk,XTest,YTest,LTest,nchunks] = load_data(db_name)
%
% This function load_data is designed to load and preprocess a dataset. 
% The function divides the dataset into chunks and prepares test data.
%
% [XChunk,YChunk,LChunk,XTest,YTest,LTest,nchunks] = load_data(db_name);
% %  Input   
%    db_name: The name of the dataset to load.
% %  Output 
%    XChunk: A cell array of chunks of the image features.
%    YChunk: A cell array of chunks of the text features.
%    LChunk: A cell array of chunks of the labels.
%    XTest: Test image features.
%    YTest: Test text features.
%    LTest: Test labels.
%    nchunks: The number of chunks.
%
% % Reference:
% Online semantic embedding correlation for discrete cross-media hashing. 
% (Manuscript)
% Version1.0 -- Jan/2025
% Contant: Haoyu Hu (982258029@qq.com)
%
if strcmp(db_name, 'IAPRTC-12')
    load('IAPRTC-12.mat');
    % The amount of data loaded per batch.
    chunksize = 2000;
    clear V_tr V_te
    %The number of anchor points set during the kernel process for the dataset.
    nXanchors = 1500; nYanchors = 1500;
    anchor_idx = randsample(size(I_tr ,1), nXanchors);
    XAnchors = I_tr (anchor_idx,:);
    anchor_idx = randsample(size(T_tr,1), nYanchors);
    YAnchors = T_tr(anchor_idx,:);
    [I_tr1,I_te1]=Kernel_Feature(I_tr,I_te,XAnchors);
    [T_tr1,T_te1]=Kernel_Feature(T_tr,T_te,YAnchors);
    % Concatenate training and testing data
    X = [I_tr1; I_te1]; Y = [T_tr1; T_te1]; L = [L_tr; L_te];
    % Randomly shuffle the dataset index
    R = randperm(size(L,1));
    % Divide the dataset into query and sample indexes
    queryInds = R(1:2000);
    sampleInds = R(2001:end);
    % Calculate the number of chunks
    nchunks = floor(length(sampleInds)/chunksize);  
    % Initialize cell arrays for chunks
    XChunk = cell(nchunks,1);
    YChunk = cell(nchunks,1);
    LChunk = cell(nchunks,1);
    % Divide the dataset into chunks
    for subi = 1:nchunks-1
        XChunk{subi,1} = X(sampleInds(chunksize*(subi-1)+1:chunksize*subi),:);
        YChunk{subi,1} = Y(sampleInds(chunksize*(subi-1)+1:chunksize*subi),:);
        LChunk{subi,1} = L(sampleInds(chunksize*(subi-1)+1:chunksize*subi),:);
    end
    XChunk{nchunks,1} = X(sampleInds(chunksize*subi+1:end),:);
    YChunk{nchunks,1} = Y(sampleInds(chunksize*subi+1:end),:);
    LChunk{nchunks,1} = L(sampleInds(chunksize*subi+1:end),:);
    % Prepare test data
    XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
    % Clear unnecessary variables
    clear X Y L
end
