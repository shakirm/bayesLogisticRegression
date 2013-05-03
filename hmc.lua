-- Hybrid Monte Carlo Sampling
require 'bayesLogReg'

function hmc(initVec,X,y,alpha, numSamples,maxLeaps,stepSize,display)

	local D = X:size(1);
	local N = X:size(2);
	
	local omega = initVec:clone()
	local nParams = omega:size(2); -- 1xD+1

	-- Initial eval for BLR
	local logp, grad = bayesLogReg(initVec,X,y,alpha);
	
	local samples = torch.Tensor(numSamples+1,nParams):zero();
	samples[{{1},{}}] = omega;
	
	for i = 1,numSamples do
		local p = torch.randn(nParams,1); -- momentum
		local H = 0.5*torch.dot(p,p) - logp; -- Hamiltonian
		
		local omegaNew = omega:clone();
		local gradNew = grad:clone();
		
		local numLeaps = torch.squeeze(torch.ceil(torch.rand(1)*maxLeaps)) -- randomise num leapfrog steps
		-- Do leapfrog discretisation
		for t = 1,numLeaps do
			p = p + torch.mul(gradNew,0.5*stepSize) -- half step in p
			omegaNew = omegaNew + torch.mul(p,stepSize); -- step in p
			logpNew, gradNew = bayesLogReg(omegaNew,X,y,alpha); -- update gradient
			p = p + torch.mul(gradNew,0.5*stepSize); -- half step in p;
		end;
		
		local accept = 0;
		Hnew = 0.5*torch.dot(p,p) - logpNew; -- New Hamiltonian
		dH = Hnew - H;
		accprob = torch.exp(-dH);
		if accprob > 1 then
			accprob = 1;
		end;
		
		local accept = 0;
		if (torch.squeeze(torch.rand(1)) < accprob) then
			accept = 1;
			grad = gradNew:clone();
			logp = logpNew
			omega = omegaNew:clone();
		end;
		samples[{{i+1},{}}] = omega:clone();
		
		print(i, 'Energy',-logp,'Accept',accept,'Leapfrogs',numLeaps);
	end;

	return samples
end;




