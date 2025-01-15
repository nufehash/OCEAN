function [precision, recall] = precision_recall(ids, Lbase, Lquery)
%
%This function precision_recall calculates the precision and recall curves for a given set of query results. 
% Precision and recall are common metrics used to evaluate the performance of information retrieval systems.
%[precision, recall] = precision_recall(ids, Lbase, Lquery);
  %Input
  % ids: A matrix where each column represents the ranked list of indices of the base dataset for a particular query.
  % Lbase: A matrix of labels for the base dataset.
  % Lquery: A matrix of labels for the query dataset.
  % Output
  % precision: A vector representing the precision values at different recall levels.
  % recall: A vector representing the recall levels.
  %
%%
% % Reference:
% Online semantic embedding correlation for discrete cross-media hashing. 
% (Manuscript)
% Version1.0 -- Jan/2025
% Contant: Haoyu Hu (982258029@qq.com)
%
% Initialize variables
nquery = size(ids, 2);
K = size(ids, 1);
P = zeros(K, nquery);
R = zeros(K, nquery);
% Calculate precision and recall for each query
for i = 1 : nquery
    label = Lquery(i, :);
    label(label == 0) = -1;
    idx = ids(:, i);
    imatch = sum(bsxfun(@eq, Lbase(idx(1:K), :), label), 2) > 0;
    LK = sum(imatch);
    if LK == 0
        continue;
    end
    Lk = cumsum(imatch);
    P(:, i) = Lk ./ (1:K)';
    R(:, i) = Lk ./ LK;
end
% Calculate the mean precision and recall
mP = mean(P, 2);
mR = mean(R, 2);
mP = [mP(1); mP];
mR = [0; mR];
% Interpolate precision values for a range of recall levels
recall = (0.0:0.001:max(mR))';
precision = interpolate_pr(mR, mP, recall)';

end


function precision = interpolate_pr(r, p, recs)
%
% Input
% r: Recall values.
% p: Precision values.
% recs: Recall levels for which to interpolate precision values.
% Output
% precision: Interpolated precision values.
%
% Check input sizes
n = numel(p);
if (n ~= numel(r))
    error('two first arguments should be of the same size');
end
%Interpolate precision for each recall level
for j = 1:numel(recs)
    rec = recs(j);
    done = 0;
    for i = 1:n-1
        if (r(i) <= rec && rec <= r(i+1))
            done = 1;
            if (r(i) == r(i+1))
                precision(j) = (p(i) + p(i+1)) / 2;
            else
                precision(j) = p(i) + (rec - r(i)) * (p(i+1) - p(i)) / (r(i+1) - r(i));
            end
            break;
        end
    end
    
    if ~done
        error('not done for %.2f!', rec);
    end
end

end
