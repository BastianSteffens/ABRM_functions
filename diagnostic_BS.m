function Match = diagnostic_BS(x)
%startup
%mrstModule add diagnostics;

%% Set up and solve flow problem
% To illustrate the various concepts, we use a rectangular reservoir with
% 2 wells, 1 injectors an 1 producers

% This is the only parameter that we are varying around in the first test
% run of PSO. The idea here is to get the "right" model answer.
truth = x;
% Grid
nx = 10;
ny = 10;
nz = 10;
G = cartGrid([nx,ny,nz],[1000,1000,100]);
G = computeGeometry(G);

% Petrophysical data

p = 0.25;
K1 = 10;
K2 = 1000;

rock = makeRock(G, K1(:), p(:));
rock.perm(1:truth) = K1;
rock.perm(truth+1:end) = K2;

hT  = computeTrans(G, rock);
pv  = sum(poreVolume(G,rock));

% Fluid model
gravity reset off
fluid = initSingleFluid('mu', 1.0*(centi)*poise, 'rho', 1014.0*(kilogram/meter)^3.0);
%fluid = initSingleFluid('mu',1, 'rho',1014);
% Wells
cellInx_I1 = (1:nx*ny:nx*ny*nz);
cellInx_P1 = (nx*ny:nx*ny:nx*ny*nz);
n = 2;
W = addWell([],  G, rock, cellInx_I1, ...
    'Type', 'rate', 'Comp_i', 1, 'name', 'I1', 'Val', pv/4);

W = addWell(W, G, rock, cellInx_P1, ...
    'Type','rate',  'Comp_i', 0, 'name', 'P1', 'Val', -pv/4);

% Initial reservoir state
state = initState(G, W, 0.0, 1.0);

%% Compute basic quantities
% To compute the basic quantities, we need a flow field, which we obtain by
% solving a single-phase flow problem. Using this flow field, we can
% compute time-of-flight and numerical tracer partitions. For comparison,
% we will also tracer streamlines in the same flow field.

state = incompTPFA(state, G, hT, fluid, 'wells', W);
D = computeTOFandTracer(state, G, rock, 'wells', W);

%% F-Phi diagram
% To define a measure of dynamic heterogeneity, we can think of the
% reservoir as a bundle of non-coummunicating volumetric flow paths
% (streamtubes) that each has a volume, a flow rate, and a residence time.
% For a given time, the storage capacity Phi, is the fraction of flow paths
% in which fluids have reached the outlet, whereas F represent the
% corresponding fractional flow. Both are monotone functions of residence
% time, and by plotting F versus Phi, we can get a visual picture of the
% dynamic heterogeneity of the problem. In a completely homogeneous
% displacement, all flowpaths will break through at the same time and hence
% F(Phi) is a straight line from (0,0) to (1,1). In a heterogeneous
% displacement, F(Phi) will be a concave function in which the steep
% initial slope corresponds to high-flow regions giving early breakthrough
% and, whereas the flat trailing tail corresponds to low-flow and stagnant
% regions that would only break through after very long time
[F,Phi] = computeFandPhi(poreVolume(G,rock), D.tof);


%% Lorenz coefficient
% The further the F(Phi) curve is from a straight line, the larger the
% difference between fast and slow flow paths. The Lorenz coefficient
% measures the dynamic heterogeneity as two times the area between F(Phi)
% and the straight line F=Phi.
%This is what we want to match / use as metric to compare our model with
%the PSO generated model.
%the 0.55 etc is the value I lorenz coefficient I get for truth = 500.
Match = abs(0.55 - computeLorenz(F,Phi));
%Match = computeLorenz(F,Phi);

%% Copyright notice

% <html>
% <p><font size="-1">
% Copyright 2009-2018 SINTEF Digital, Mathematics & Cybernetics.
% </font></p>
% <p><font size="-1">
% This file is part of The MATLAB Reservoir Simulation Toolbox (MRST).
% </font></p>
% <p><font size="-1">
% MRST is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% </font></p>
% <p><font size="-1">
% MRST is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% </font></p>
% <p><font size="-1">
% You should have received a copy of the GNU General Public License
% along with MRST.  If not, see
% <a href="http://www.gnu.org/licenses/">http://www.gnu.org/licenses</a>.
% </font></p>
% </html>
end