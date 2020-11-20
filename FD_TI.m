function [FD_performance] = FD_TI(x)
    mrstModule add diagnostics;
    mrstModule add incomp;
    mrstModule add deckformat;

    %disp('Loading Data files into Matlab and running Flow Diagnostics')
   
    %% Load File into MRST 

    ModelName = 'TI%s.DATA';
    ModelIndex = string(x);
    ModelNameIndex = sprintf(ModelName,ModelIndex);

    current_dir = 'C:\AgentBased_RM\Output\training_images\current_run\DATA\';
    fn = fullfile(current_dir, ModelNameIndex);
    deck = readEclipseDeck(fn);
    deck = convertDeckUnits(deck);

    % Load Grid
    G = initEclipseGrid(deck);
    G = computeGeometry(G);

    % Petrophysics
    rock  = initEclipseRock(deck);
    rock  = compressRock(rock, G.cells.indexMap);
    hT  = computeTrans(G, rock);
    pv  = sum(poreVolume(G,rock));
  
    % Fluid model
    gravity reset off
    fluid = initSingleFluid('mu', 1*centi*poise, 'rho', 1014*kilogram/meter^3);
    
    % well position
    W = processWells(G, rock, deck.SCHEDULE.control(1));

    % Initial reservoir state (remove last value (1.0) form inistate. maybe
    % mistake. hceck that later. saw that its also don in other
    % example simulations.
    state = initState(G, W, 0);

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
    LC = computeLorenz(F,Phi);

    %% Sweep effciency diagram
    % We can also define measures of sweep efficiency that tell how effective
    % injected fluids are being used. The volumetric sweep efficiency Ev is
    % defined as the ratio of the volume that has been contacted by the
    % displacing fluid at time t and the volume contacted at infinite time.
    % This quantity is usually related to dimensionless time td=dPhi/dF
    [Ev,tD] = computeSweep(F,Phi);
    
    % sometimes Ev and tD are missing 10 values in the end. its basicallz just
    % several 1.0 that are missing. so will just add them so that reshaping
    % wont be a problem within python.

    if not(length(Ev)==length(F))
        values_to_add = length(F) - length(Ev);
        Ev(end + values_to_add) = 1;
        tD(end + values_to_add) = 1;
    end
        
    Ev(end)= [];
    tD(end)= [];
    F(end)= [];
    Phi(end)= [];
    LC_long = zeros(length(Ev),1) + LC;
    tof = D.tof(:,2);
    FD_performance = [Ev;tD;F;Phi;LC_long;tof];
end