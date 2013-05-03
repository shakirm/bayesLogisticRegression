-- BLR log probability
require 'blr_util'

function bayesLogReg(w, X, y, alpha)
	-- Data must be DxN
	local D = X:size(1);
	local N = X:size(2);
	local m = torch.Tensor(1,D):zero();

	local eta = torch.mm(w,X) --X:t(),w:t())

	-- Compute log-probability
	local Term = torch.Tensor(3):zero();
	Term[1] = normpdfln(w,m,alpha);
	Term[2] = torch.dot(y,eta);
	Term[3] = -torch.sum(llp(eta));
	local logp = torch.sum(Term);

	-- Compute gradient
	local diff = y - sigmoid(eta)
	local dw = torch.mm(diff,X:t())
	dw = dw - torch.mul(w,1/alpha);
	local grad = dw;

	return logp, grad;
end;