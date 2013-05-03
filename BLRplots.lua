
local samples = torch.load('blr_samples.dat');


-- Posterior mean
print('Posterior Mean')
print(torch.mean(samples,1));

gnuplot.closeall()

-- MCMC Trace Plots
gnuplot.figure();
gnuplot.plot(torch.squeeze(samples[{{},{1}}]),'+-')

gnuplot.figure();
gnuplot.plot(torch.squeeze(samples[{{},{2}}]),'+-')

gnuplot.figure();
gnuplot.plot(torch.squeeze(samples[{{},{3}}]),'+-')

-- Density plots
gnuplot.figure();
gnuplot.hist(torch.squeeze(samples[{{},{1}}]),10)

gnuplot.figure();
gnuplot.hist(torch.squeeze(samples[{{},{2}}]),10)

gnuplot.figure();
gnuplot.hist(torch.squeeze(samples[{{},{3}}]),10)

