function eva = evaluate_OCEAN(XChunk,YChunk,LChunk,XTest,YTest,LTest,param)
% %
% This function is designed to evaluate the performance. 
% The function processes the dataset in chunks, trains the model, and evaluates the retrieval performance using mean Average Precision. 
% 
% eva = evaluate_OCEAN(XChunk,YChunk,LChunk,XTest,YTest,LTest,param);
  % Input   
  % XChunk: A cell array of chunks of the image features.
  % YChunk: A cell array of chunks of the text features.
  % LChunk: A cell array of chunks of the labels.
  % XTest: Test image features.
  % YTest: Test text features.
  % LTest: Test labels.
  % param: A structure containing parameters.
  % Output
  % eva: A cell array containing evaluation results for each chunk
  %
  %
% Reference:
% Online semantic embedding correlation for discrete cross-media hashing. 
% (Manuscript)
% Version1.0 -- Jan/2025
%
%   Initialize evaluation results
    eva = cell(param.nchunks,1);
    param.xi=1e0;
    param.delta=1e-1;
    param.max_iter=5;
%   Loop through each chunk
    for chunki = 1:param.nchunks
        fprintf('-----chunk----- %3d\n', chunki);
        LTrain = cell2mat(LChunk(1:chunki,:));
        XTrain_new = XChunk{chunki,:};
        YTrain_new = YChunk{chunki,:};
        LTrain_new = LChunk{chunki,:};
        GTrain_new = NormalizeFea(LTrain_new,1); 
        % Hash code learning
        % First chunk
        if chunki == 1
            tic;
            [BB,XW,YW,HH] = train_OCEAN0(XTrain_new,YTrain_new,LTrain_new,GTrain_new,param);
            % Training Time
            traintime=toc; 
            fprintf('The training time for the %d-th round is==%6.4fs \n', chunki,traintime);
            evaluation_info.trainT=traintime;
        % Subsequent chunks
        else
            tic;
            [BB,XW,YW,HH] = train_OCEAN(XTrain_new,YTrain_new,LTrain_new,GTrain_new,BB,HH,param);
            % Training Time
            traintime=toc;  
            fprintf('The training time for the %d-th round is==%6.4fs \n', chunki,traintime);
            evaluation_info.trainT=traintime;
        end
        tic;
       % Test data compression
        BxTest = compactbit(XTest*XW>0);
        ByTest = compactbit(YTest*YW>0);
        evaluation_info.compressT=toc;
        % Training data compression
        B = cell2mat(BB(1:end,:));
        BxTrain = compactbit(B>0);
        ByTrain = BxTrain;
        tic;
        % After processing 9 batches of data, the evaluation of retrieval performance (MAP) began.
        if chunki == 9
            DHamm = hammingDist(BxTest, BxTrain);
            [~, orderH] = sort(DHamm, 2);
            %% iamge as query to retrieve text database
            evaluation_info.Image_VS_Text_MAP = mAP(orderH', LTrain, LTest);
            DHamm = hammingDist(ByTest, ByTrain);
            [~, orderH] = sort(DHamm, 2);
           %% text as query to retrieve image database
            evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
        end
        DHamm = hammingDist(BxTest, BxTrain);
        [~, orderH] = sort(DHamm, 2);
        evaluation_info.Image_VS_Text_MAP = mAP(orderH', LTrain, LTest);
        DHamm = hammingDist(ByTest, ByTrain);
        [~, orderH] = sort(DHamm, 2);
        evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
        evaluation_info.testT=toc;   
        eva{chunki} = evaluation_info;
        clear evaluation_info
        
    end
end
