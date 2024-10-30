function [XChunk,YChunk,LChunk,XTest,YTest,LTest,nchunks] = load_data(db_name)
if strcmp(db_name, 'IAPRTC-12')
    load('IAPRTC-12.mat');
    chunksize = 2000;
    clear V_tr V_te
    nXanchors = 1500; nYanchors = 1500;
    anchor_idx = randsample(size(I_tr ,1), nXanchors);
    XAnchors = I_tr (anchor_idx,:);
    anchor_idx = randsample(size(T_tr,1), nYanchors);
    YAnchors = T_tr(anchor_idx,:);
    [I_tr1,I_te1]=Kernel_Feature(I_tr,I_te,XAnchors);
    [T_tr1,T_te1]=Kernel_Feature(T_tr,T_te,YAnchors);
    X = [I_tr1; I_te1]; Y = [T_tr1; T_te1]; L = [L_tr; L_te];
    R = randperm(size(L,1));
    queryInds = R(1:2000);
    sampleInds = R(2001:end);
    nchunks = floor(length(sampleInds)/chunksize);    
    XChunk = cell(nchunks,1);
    YChunk = cell(nchunks,1);
    LChunk = cell(nchunks,1);
    for subi = 1:nchunks-1
        XChunk{subi,1} = X(sampleInds(chunksize*(subi-1)+1:chunksize*subi),:);
        YChunk{subi,1} = Y(sampleInds(chunksize*(subi-1)+1:chunksize*subi),:);
        LChunk{subi,1} = L(sampleInds(chunksize*(subi-1)+1:chunksize*subi),:);
    end
    XChunk{nchunks,1} = X(sampleInds(chunksize*subi+1:end),:);
    YChunk{nchunks,1} = Y(sampleInds(chunksize*subi+1:end),:);
    LChunk{nchunks,1} = L(sampleInds(chunksize*subi+1:end),:);
    XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
    clear X Y L
end
