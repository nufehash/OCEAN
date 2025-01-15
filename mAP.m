function map = mAP(ids, Lbase, Lquery)
% 
% This function mAP calculates the mean Average Precision for a given set of query results.
% map = mAP(ids, Lbase, Lquery);
% %  Input  
% ids: A matrix where each column represents the ranked list of indices of the base dataset for a particular query.
% Lbase: A matrix of labels for the base dataset, where each row corresponds to a sample and each column corresponds to a label.
% Lquery: A matrix of labels for the query dataset, where each row corresponds to a query and each column corresponds to a label.
% %  Output 
% map: The mean Average Precision (mAP) of the retrieval system.
% 
% % Reference:
% Online semantic embedding correlation for discrete cross-media hashing. 
% (Manuscript)
% Version1.0 -- Jan/2025
% Contant: Haoyu Hu (982258029@qq.com)
%
%%
% Initialize variables
nquery = size(ids, 2);
APx = zeros(nquery, 1);
R = size(Lbase,1); % Configurable
% Loop through each query
for i = 1 : nquery
    label = Lquery(i, :);
    label(label == 0) = -1;
    idx = ids(:, i);
    imatch = sum(bsxfun(@eq, Lbase(idx(1:R), :), label), 2) > 0;
    LX = sum(imatch);
    Lx = cumsum(imatch);
    Px = Lx ./ (1:R)';
    if LX ~= 0
        APx(i) = sum(Px .* imatch) / LX;
    end
end
% Calculate the mean Average Precision
map = mean(APx);
%
end
