function [vectorized__data] = vectorize_data(X)

% This function vectorizes the lower triangular matrix of the full data.
% No the embedded data.
    for i = 1:1:240
        x = X(:,:,i);
        mask = tril(true(size(x)),-1);
        out = x(mask);
   
        vec_X(i,:) = out;
    end

    vectorized__data = vec_X;
end