-- regression
require "bayesLogReg"
require "blr_util"
require "csv"
require "hmc"

torch.manualSeed(1)

-- Extract and prepare data 
local raw = csv.load{path='ripley.csv',mode='raw',header=0};
local u = torch.Tensor(raw)
local XX = u[{{},{1,2}}]
local D = XX:size(2);
local N = XX:size(1);
local y = u[{{},{3}}]
y = y:t();

-- Standardise data
local mu = torch.mean(XX,1);
local s = torch.std(XX,1);
print(XX:size())
print(mu)
print(s)
XX = torch.cdiv(XX - torch.repeatTensor(mu,N,1), torch.repeatTensor(s,N,1));

local X = torch.Tensor(D+1,N):fill(1);
X[{{2,D+1},{}}] = XX:t(); -- add bias to data

-- Initialise parameters
--local w = torch.Tensor(1,D+1):fill(0);
local w = torch.randn(1,D+1);
local alpha = 0.5;

-- Sampler settings
local numSamples = 200;
local nLeaps = 200;
local stepSize = 0.1;
local display = 1;

-- Run Hybrid Monte Carlo Sampler
local samples, stats = hmc(w,X,y,alpha, numSamples,nLeaps,stepSize,display)
print(samples[{{numSamples},{}}])

-- Save result of data
torch.save('blr_samples.dat',samples);

--[[
--local w = torch.randn(1,D+1);
local w = torch.Tensor(1,D+1):fill(1);
local y = torch.Tensor(1,N):fill(1);
local alpha = 0.5;
local X = torch.Tensor(D+1,N):fill(1);
--X[{{2,D+1},{}}] = torch.randn(D,N); -- add bias to data 
--]]