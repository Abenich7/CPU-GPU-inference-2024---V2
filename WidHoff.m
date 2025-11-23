function [W,b,EpochErr]=WidHoff(X,T,alpha,NumEpochs)
% function [W,b,EpochErr]=WidHoff(X,T,alpha,NumEpochs)
% Data in X should be dimension x num of points
% Target data should be dimension x num of points
% OUTPUT: Weight matrix W is dim(T) x dim(X), and b is dim(T)

% Some error checks to be sure data is input correctly and initializations:
[rX,cX]=size(X);
[rT,cT]=size(T);
if cX~=cT
error('Error in inputs: Number of points do not match.\n');
end
NumPoints=cX;
W=randn(rT,rX);
b=randn(rT,1);
EpochErr=zeros(NumEpochs,1);
% Main Code:
for k=1:NumEpochs
idx=randperm(NumPoints);
for j=1:NumPoints
ThisOut=W*X(:,idx(j))+b;
ThisErr=T(:,idx(j))-ThisOut;
%Update the weights and biases using Widrow-Hoff:
W=W+alpha*ThisErr*X(:,idx(j))';
b=b+alpha*ThisErr;
end
EpochErr(k)=norm((W*X+b*ones(1,NumPoints))-T);
end

