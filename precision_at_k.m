function precision = precision_at_k(ids, Lbase, Lquery, K)
%
% This function precision_at_k calculates the precision at k (P@K) for a given set of query results.
% P@K is a measure of the relevance of the top k items in a ranked list of results. 
%
%%precision = precision_at_k(ids, Lbase, Lquery, K);
%Input
% ids: A matrix where each column represents the ranked list of indices of the base dataset for a particular query.
% Lbase: A matrix of labels for the base dataset.
% Lquery: A matrix of labels for the query dataset.
% K: The number of top items to consider for precision calculation.
% Output
% precision: A vector where each element represents the precision at k for each query.
%% % Reference:
% Online semantic embedding correlation for discrete cross-media hashing. 
% (Manuscript)
% Version1.0 -- Jan/2025
% Contant: Haoyu Hu (982258029@qq.com)
%
%
% Check if K is provided
if ~exist('K','var')
	K = size(Lbase,1);
end
% Initialize variables
nquery = size(ids, 2);
P = zeros(K, nquery);
% Calculate precision for each query
for i = 1 : nquery
    label = Lquery(i, :);
    label(label == 0) = -1;
    idx = ids(:, i);
    imatch = sum(bsxfun(@eq, Lbase(idx(1:K), :), label), 2) > 0;
    Lk = cumsum(imatch);
    P(:, i) = Lk ./ (1:K)';
end
% Calculate the mean precision for each query
precision = mean(P, 2);
end
