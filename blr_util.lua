-- Commonly used utility functions
-- SHakir, April 2013

-- Compute sigmoid function: 1/1+exp(-x)
function sigmoid(x)
	
	local eta = torch.exp(-x);
	local val = torch.pow(eta:add(1),-1); 
	
	return val;
end;

-- Compute logistic log-partition function: log(1+exp(x))
function llp(x)
	
	local val = torch.log(torch.add(torch.exp(x),1));
	
	return val;
end;

-- Compute PDF of a Gaussian
function normpdf(x, m, S)
	-- take input, mean m, variance s
	return torch.exp(normpdfln(x,m,S));
end;

-- Compute log of Gaussian pdf
function normpdfln(x,m,v)
	-- take input, mean m, variance s
	-- x and m must be row vector
	
	local D = x:size(2);
	local t1 = torch.Tensor(D):fill(1);
	t1:mul(0.5*math.log(2*math.pi*v)); -- 0.5*log(2*pi)
	local t2 = torch.pow(torch.add(x,-m),2); -- (x-m)^2
	t2:div(2*v); -- t2/(2s)
	
	val = torch.sum(torch.add(-t1,-t2))
	return val;
end;	






