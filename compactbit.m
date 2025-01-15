function cb = compactbit(b)
%
% This function compactbit is designed to compact an array of bits into a more space-efficient representation. 
%
% cb = compactbit(b)?
  % Input
  % b = bits array, where each row represents a sample and each column represents a bit.
  % Output
  % cb = compacted string of bits (using words of 'word' bits), where each bit is represented using 8-bit words (uint8).
  %
%%
% % Reference:
% Online semantic embedding correlation for discrete cross-media hashing. 
% (Manuscript)
% Version1.0 -- Jan/2025
% Contant: Haoyu Hu (982258029@qq.com)
%

%
% Get the size of the input bit array
[nSamples nbits] = size(b);
% Calculate the number of 8-bit words needed
nwords = ceil(nbits/8);
% Initialize the compacted bit array
cb = zeros([nSamples nwords], 'uint8');
% Compact the bits into 8-bit words
for j = 1:nbits
    w = ceil(j/8);
    cb(:,w) = bitset(cb(:,w), mod(j-1,8)+1, b(:,j));
end
%
end
