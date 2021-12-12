clear
clc
close all
%% Set up Matlab for Flow Diagnostics

addpath(genpath('C:\AgentBased_RM'));
addpath(genpath('C:\AgentBased_RM\Functions'))
addpath('C:\AgentBased_RM\MRST\mrst-2019');

%%
run('C:\AgentBased_RM\MRST\mrst-2019\startup.m')

clc
close all
mrstModule clear
mrstModule add ad-props ad-core ad-blackoil  blackoil-sequential diagnostics mrst-gui  incomp coarsegrid
