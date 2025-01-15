function fea = NormalizeFea(fea,row)
% This function NormalizeFea normalizes the features in a matrix fea to have unit norm. 
% The function can normalize either each row or each column of the matrix, depending on the value of the row parameter.
% fea = NormalizeFea(fea,row);
% % Input   
% fea: The feature matrix to be normalized.
% row: A flag indicating whether to normalize rows (1) or columns (0). Default is 1.
% if row == 1, normalize each row of fea to have unit norm;
% if row == 0, normalize each column of fea to have unit norm;
% % Output
% fea is the normalized feature matrix
%%%
% % Reference:
% Online semantic embedding correlation for discrete cross-media hashing. 
% (Manuscript)
% Version1.0 -- Jan/2025
% Contant: Haoyu Hu (982258029@qq.com)
%
%
% Check if row is provided
if ~exist('row','var')
    row = 1;
end
% Normalize rows
if row
    nSmp = size(fea,1);
    feaNorm = max(1e-14,full(sum(fea.^2,2)));
    fea = spdiags(feaNorm.^-.5,0,nSmp,nSmp)*fea;
% Normalize columns
else
    nSmp = size(fea,2);
    feaNorm = max(1e-14,full(sum(fea.^2,1))');
    fea = fea*spdiags(feaNorm.^-.5,0,nSmp,nSmp);
end
return;
