close all; clear; clc;
db = {'IAPRTC-12'};%%  load dataset
hashmethods = {'OCEAN'};
%%  initialization
param.nbits = 16;   %% Hash codes length setting
param.top_K = 2000; %% The top K(2000)results with the highest similarity.
[XChunk,YChunk,LChunk,XTest,YTest,LTest,chunks] = load_data(db);%% Load training dataset
fprintf('========%s %d bits start======== \n', 'OCEAN',param.nbits);
OCEANparam = param;
OCEANparam.nchunks=chunks;%% Number of data chunks
%% Parameter setting
OCEANparam.theta=0.5;
OCEANparam.alpha=1e4;
OCEANparam.beta=1e0;
OCEANparam.mu=1e5;
OCEANparam.omega=1e0;
%%
%% --------------------OURS----------------------------%%
eva_info_ = evaluate_OCEAN(XChunk,YChunk,LChunk,XTest,YTest,LTest,OCEANparam);
fprintf('Image_VS_Text_MAP: %6.4f.\n', eva_info_{9}.Image_VS_Text_MAP);
fprintf('Text_VS_Image_MAP: %6.4f.\n', eva_info_{9}.Text_VS_Image_MAP);

