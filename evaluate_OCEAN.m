function eva = evaluate_OCEAN(XChunk,YChunk,LChunk,XTest,YTest,LTest,param)
    
    eva = cell(param.nchunks,1);
    param.xi=1e0;
    param.delta=1e-1;
    param.max_iter=5;
    param.omega=1e0;
    for chunki = 1:param.nchunks
        fprintf('-----chunk----- %3d\n', chunki);
        LTrain = cell2mat(LChunk(1:chunki,:));
        XTrain_new = XChunk{chunki,:};
        YTrain_new = YChunk{chunki,:};
        LTrain_new = LChunk{chunki,:};
        GTrain_new = NormalizeFea(LTrain_new,1); 
        % Hash code learning
        if chunki == 1
            tic;
            [BB,XW,YW,HH] = train_OCEAN0(XTrain_new,YTrain_new,LTrain_new,GTrain_new,param);
            traintime=toc;  % Training Time
            fprintf('The training time for the %d-th round is==%6.4fs \n', chunki,traintime);
            evaluation_info.trainT=traintime;
        else
            tic;
            [BB,XW,YW,HH] = train_OCEAN(XTrain_new,YTrain_new,LTrain_new,GTrain_new,BB,HH,param);
            traintime=toc;  % Training Time
            fprintf('The training time for the %d-th round is==%6.4fs \n', chunki,traintime);
            evaluation_info.trainT=traintime;
        end
        tic;
        BxTest = compactbit(XTest*XW>0);
        ByTest = compactbit(YTest*YW>0);
        evaluation_info.compressT=toc;
        B = cell2mat(BB(1:end,:));
        BxTrain = compactbit(B>0);
        ByTrain = BxTrain;
        tic;
        if chunki == 9
            DHamm = hammingDist(BxTest, BxTrain);
            [~, orderH] = sort(DHamm, 2);
            evaluation_info.Image_VS_Text_MAP = mAP(orderH', LTrain, LTest);
            evaluation_info.Image_VS_Text_MAP100 = mAP(orderH', LTrain, LTest, 100);
            DHamm = hammingDist(ByTest, ByTrain);
            [~, orderH] = sort(DHamm, 2);
            evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
            evaluation_info.Text_VS_Image_MAP100 = mAP(orderH', LTrain, LTest, 100);
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
