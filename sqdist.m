function d=sqdist(a,b)
% The sqdist function efficiently computes the squared Euclidean distance matrix between two sets of points. 
% sqdist - computes squared Euclidean distance matrix; computes a rectangular matrix of pairwise distances
% between points in a (given in columns) and points in b
% d=sqdist(a,b);
% Input
% a: A matrix where each column represents a point in a multi-dimensional space.
% b: A matrix where each column represents a point in the same multi-dimensional space.
% Output
% d: A matrix where each element d(i, j) represents the squared Euclidean distance between the i-th point in a and the j-th point in b.
% 
%%
%%Reference:
% Online semantic embedding correlation for discrete cross-media hashing. 
% (Manuscript)
% Version1.0 -- Jan/2025
% Contant: Haoyu Hu (982258029@qq.com)
%%
% 
% Calculate the squared norms of the points in a and b:
aa = sum(a.^2,1); 
bb = sum(b.^2,1);
% Calculate the dot product of a and b
ab = a'*b; 
% Compute the squared Euclidean distance matrix
d = (repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

