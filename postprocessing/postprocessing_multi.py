
########################
import numpy as np
import pandas as pd
import os
from os import path
import bz2
import _pickle as cPickle
from scipy import interpolate
import shutil
import umap
import hdbscan
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pathlib
from skimage.util.shape import view_as_windows
from GRDECL_file_reader.GRDECL2VTK import *
from geovoronoi import voronoi_regions_from_coords
from collections import Counter
import pyvista as pv
from pyentrp import entropy as ent
from colour import Color
########################

class postprocessing_multi():
    """ Load single/multiple datasets from MOPSO and visualize/ analyze them """

    def __init__(self, data_to_process):

        # set base path where data is stored
        self.setup = dict()
        self.setup["base_path"] = pathlib.Path(__file__).parent

        # date & time at what dataset is created needs to be fed in here as string
        self.data_to_process = data_to_process

        # how many different datasets:
        self.n_datasets = len(self.data_to_process)

        # data storage
        self.df_position = pd.DataFrame()
        self.df_performance = pd.DataFrame()
        self.df_tof = pd.DataFrame()
        self.setup_all = dict()
        self.FD_targets = dict()
        self.Phi_interpolated = []
        self.F_interpolated = []
        self.LC_interpolated = []

    def read_data(self):
        """ read individual datasets and conconate them to one big df """

        for i in range(0,self.n_datasets):
            path = str(self.setup["base_path"] / "../../Output/")+ "/"+ self.data_to_process[i] + "/"
            performance = "swarm_performance_all_iter.pbz2"
            position = "swarm_particle_values_converted_all_iter.csv"
            setup = "variable_settings.pickle"
            tof = "tof_all_iter.pbz2"

            performance_path = path + performance
            position_path = path + position
            setup_path = path + setup
            tof_path = path + tof

            # load data
            # df_performance_single = pd.read_csv(performance_path)
            df_position_single = pd.read_csv(position_path)
            
            #load compressed pickle file
            data = bz2.BZ2File(performance_path,"rb")
            df_performance_single = cPickle.load(data)
            data = bz2.BZ2File(tof_path,"rb")
            df_tof_single = cPickle.load(data)
            data = bz2.BZ2File(setup_path)
            setup_single = cPickle.load(data)


            df_position_single["dataset"] = self.data_to_process[i]
            df_performance_single["dataset"] = self.data_to_process[i]
            df_tof_single["dataset"] = self.data_to_process[i]
            
            # get F Phi curve and LC for interpolation
            Phi_points_target = setup_single["Phi_points_target"]
            F_points_target = setup_single["F_points_target"]
            
            # interpolate F-Phi curve from input points with spline
            tck = interpolate.splrep(Phi_points_target,F_points_target, s = 0)
            Phi_interpolated = np.linspace(0,1,num = len(df_performance_single.loc[(df_performance_single.iteration == 0) & (df_performance_single.particle_no == 0)]["Phi_0"]),endpoint = True)        
            F_interpolated = interpolate.splev(Phi_interpolated,tck,der = 0)
            LC_interpolated = self.compute_LC(F_interpolated,Phi_interpolated)
                    
            # Concate all data together
            self.df_position = self.df_position.append(df_position_single)
            self.df_performance = self.df_performance.append(df_performance_single)
            self.df_tof = self.df_tof.append(df_tof_single)
            self.setup_all[self.data_to_process[i]] = setup_single
            self.FD_targets[self.data_to_process[i]] = dict(Phi_interpolated = Phi_interpolated,F_interpolated =F_interpolated,LC_interpolated = LC_interpolated)

            # uniform index
            self.df_position.reset_index(drop = True,inplace = True)
            self.df_performance.reset_index(drop = True,inplace = True)
            self.df_tof.reset_index(drop = True, inplace = True)

        print("Number of models and parameters:")
        display(self.df_position.shape)
        print("Number of particles:")
        display(self.df_position.particle_no.max()+1)
        print("Number of Iterations:")
        display(self.df_position.iteration.max()+1)
        display(self.df_performance.head())
        display(self.df_position.head())
        display(self.df_tof.head())

    def get_df_best(self,misfit_tolerance,window_shape = (1,1,1),step_size = 1):
        """ create dfs that only contains the models that satisfy the misfit tolerance 
            the tof can also be upscaled by changing the window_shape and step size parameters below."""
         
        # nx = self.setup["nx"]
        # ny = self.setup["ny"]
        # nz = self.setup["nz"]
        nx = 200
        ny = 100
        nz= 7
        n_shedules = self.setup_all[self.data_to_process[0]]["n_shedules"]
        
        self.df_best_position = self.df_position.copy()
        self.df_best_performance = self.df_performance.copy()
        # get models tht fulfil misfit criterion of all shedules
        for shedule_no in range(n_shedules):
            misfit = "misfit_" + str(shedule_no)
            self.df_best_position =self.df_best_position[(self.df_best_position[misfit] <= misfit_tolerance)]
            self.df_best_performance =self.df_best_performance[(self.df_best_performance[misfit] <= misfit_tolerance)]

        if window_shape == (1,1,1) and step_size == 1:
            self.best_tof_dict = dict()
            for shedule_no in range(n_shedules):
                tof_index = "tof_" + str(shedule_no)

                df_best_tof = pd.DataFrame(columns = np.arange(int(nx/window_shape[0])*int(ny/window_shape[1])*int(nz/window_shape[2])))

                # filter out tof for all best models and make it readable for clustering
                iteration = self.df_best_position.iteration.tolist()
                particle_no =  self.df_best_position.particle_no.tolist()
                tof_all = pd.DataFrame()
                for i in range(self.df_best_position.shape[0]):

                    tof = self.df_tof[(self.df_tof.iteration == iteration[i]) & (self.df_tof.particle_no == particle_no[i])][tof_index]
                    tof.reset_index(drop=True, inplace=True)

                    tof_all = tof_all.append(tof,ignore_index = True)

                df_best_tof = tof_all
                df_best_tof["iteration"] = iteration
                df_best_tof["particle_no"] = particle_no
                df_best_tof.set_index(self.df_best_position.index.values,inplace = True)

                self.best_tof_dict[tof_index] = df_best_tof


        elif window_shape != (1,1,1):
            self.best_tof_dict = dict()
            
            for shedule_no in range(n_shedules):
                tof_index = "tof_" + str(shedule_no)
                misfit = "misfit_" + str(shedule_no)

                df_best_tof = pd.DataFrame(columns = np.arange(int(nx/window_shape[0])*int(ny/window_shape[1])*int(nz/window_shape[2])))
            
                df_best_tof_temp =self.df_tof[(self.df_tof[misfit] <= misfit_tolerance)][tof_index].copy()
                iterations = df_best_tof_temp["iteration"].unique().tolist()
                for i in range(0,len(iterations)):
                    iteration = iterations[i]
                    particle_no = df_best_tof_temp[ df_best_tof_temp.iteration == iteration].particle_no.unique().tolist()
                    for j in range(0,len(particle_no)):
                        particle = particle_no[j]
                        tof_single_particle = np.array(df_best_tof_temp[(df_best_tof_temp.iteration == iteration) & (df_best_tof_temp.particle_no == particle)][tof_index])
                        tof_single_particle_3d = tof_single_particle.reshape((nx,ny,nz))
                        tof_single_particle_moving_window = view_as_windows(tof_single_particle_3d, window_shape, step= step_size)
                        tof_single_particle_upscaled = []
                        for k in range(int(nx/window_shape[0])):
                            for l in range(int(ny/window_shape[1])):
                                for m in range(int(nz/window_shape[2])):
                                    single_cell_temp = np.round(np.mean(tof_single_particle_moving_window[k,l,m]),2)
                                    tof_single_particle_upscaled.append(single_cell_temp)
                        df_tof_single_particle_upscaled = pd.DataFrame(np.array(tof_single_particle_upscaled))
                        df_tof_single_particle_upscaled_transposed = df_tof_single_particle_upscaled.T
                        df_tof_single_particle_upscaled_transposed["particle_no"] = particle
                        df_tof_single_particle_upscaled_transposed["iteration"] = iteration

                        df_best_tof = df_best_tof.append(df_tof_single_particle_upscaled_transposed)
                
                df_best_tof.set_index(self.df_best_position.index.values,inplace = True)

                self.best_tof_dict[tof_index] = df_best_tof


    def compute_LC(self,F,Phi):
        """ Compute the Lorenz Coefficient """
        v = np.diff(Phi,1)
        
        LC = 2*(np.sum(((np.array(F[0:-1]) + np.array(F[1:]))/2*v))-0.5)
        return LC
 
    def plot_hist(self, misfit_tolerance = False):
        """ Plot histogram showing distribution for all parameters used in PSO 
            Args: filter dataset to data that are fulfill misfit tolerance """
    
        columns = self.setup_all[self.data_to_process[0]]["columns"]

        cols_range = [1,2,3]

        n_cols = int(len(cols_range))
        n_rows = int(np.ceil(len(columns)/n_cols))

        cols = cols_range* n_rows 
        len_row  = list(np.arange(1,n_rows+1,1))
        rows = sorted(n_cols*len_row)

        for i in range(0,len(rows)):
            rows[i]=rows[i].item()
            
        n_subplots = len(columns)/n_rows/n_cols

        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=(columns))

        if misfit_tolerance is True:
            
            #get rid of everzthing that isnt a parameter
            df_best_position = self.df_best_position[columns]

            for i in range(0,len(columns)):
                fig.append_trace(go.Histogram(x=df_best_position[columns[i]]),row = rows[i],col = cols[i])

                fig.update_layout(
                        showlegend=False,
                        barmode='overlay'        # Overlay both histograms
                        )
                fig.update_traces(opacity = 0.75) # Reduce opacity to see both histograms


            fig.update_layout(autosize=False,
                title= "Histogram Parameters",
                width=1000,
                height=750*(n_subplots)
            )
            fig.show()

        else:

            for i in range(0,len(columns)):

                fig.append_trace(go.Histogram(x=self.df_position[columns[i]]),row = rows[i],col = cols[i])
                fig.update_layout(
                        showlegend=False
                        )
            fig.update_layout(autosize=False,
                title= "Histogram Parameters",
                width=1000,
                height=750*(n_subplots)
            )
            fig.show()

    def plot_box(self):
        """ boxplots of how PSO parameters change over iterations """ 

        max_iters = self.df_position.iteration.max() + 1
        columns = self.setup_all[self.data_to_process[0]]["columns"]
        n_shedules = self.setup_all[self.data_to_process[0]]["n_shedules"]

        for shedule_no in range(n_shedules):
            misfit = "misfit_" + str(shedule_no)
            columns.append(misfit)
     

        for j in range(0,len(columns)):
            fig = go.Figure()
            for i in range (0,max_iters):
                fig.add_trace(go.Box(y=self.df_position[self.df_position.iteration ==i][columns[j]],name="Iteration {}".format(i)))

            fig.update_layout(
                title= columns[j],
                showlegend=False
            )
            fig.show()
    
    def plot_misfit(self,objective_0 = 0,objective_1 = 1):
        """ scatter plot showing misfit of 2 objective functions """
        # at some point can add maybe development of swarm towards pareto front
        # also highlight the pareto front

        misfit_0 = "misfit_"+ str(objective_0)
        misfit_1 = "misfit_"+ str(objective_1)

        fig = make_subplots(rows =1, cols = 1,)

        fig.add_trace(go.Scatter(x = self.df_position[misfit_0], y=self.df_position[misfit_1],
                                mode='markers',
                                line = dict(color = "lightgray"),
                                name='Simulated'),row =1, col =1)
        fig.add_trace(go.Scatter( x= self.df_best_position[misfit_0],y=self.df_best_position[misfit_1],
                                    mode = "markers",
                                line = dict(color = "magenta")),row =1, col =1)

        fig.show()


    def plot_performance(self,shedule_no = 0):

        """ Create plots showing the Flow Diagnostic Performance of the PSO over n iterations
            Args: filter dataset to data that are fulfill misfit tolerance """

        # Create traces
        LC = "LC_" + str(shedule_no)
        misfit = "misfit_" + str(shedule_no)
        Phi = "Phi_" + str(shedule_no)
        F = "F_" + str(shedule_no)
        EV_index = "EV_" + str(shedule_no)
        tD_index = "tD_" + str(shedule_no)

        fig = make_subplots(rows = 2, cols = 2,
                        subplot_titles = ("Misfit","LC plot","F - Phi Graph","Sweep Efficieny Graph"))

        ### Misfit ###
        
        fig.add_trace(go.Scatter(x = self.df_position.index[(self.df_position.dataset == self.data_to_process[0])], y=self.df_position[misfit][(self.df_position.dataset == self.data_to_process[0])],
                                mode='markers',
                                line = dict(color = "black"),
                                name='misfit'),row =1, col =1)
        fig.add_trace(go.Scatter( x= self.df_best_position.index[(self.df_best_position.dataset == self.data_to_process[0])],y=self.df_best_position.loc[(self.df_best_position.dataset == self.data_to_process[0]),misfit],
                                mode = "markers",
                                line = dict(color = "magenta")))
        fig.update_xaxes(range = [0,self.df_position.index.max()],row =1, col =1)
        fig.update_yaxes(range = [0,1], row =1, col = 1)

        ### LC plot ###
        
        fig.add_trace(go.Scatter(x = self.df_position.index, y=self.df_position[LC],
                                mode='markers',
                                line = dict(color = "lightgray"),
                                name='Simulated'),row =1, col =2)
        fig.add_trace(go.Scatter( x= self.df_best_position.index,y=self.df_best_position[LC],
                                    mode = "markers",
                                line = dict(color = "magenta")),row =1, col =2)
        fig.add_shape(
                # Line Horizontal
                    type="line",
                    x0=0,
                    y0=self.FD_targets[self.data_to_process[0]]["LC_interpolated"], # make date a criterion taht can be changed
                    x1=self.df_position.index.max(),
                    y1=self.FD_targets[self.data_to_process[0]]["LC_interpolated"],
                    line=dict(
                        color="red",
                        width=2),row =1, col = 2)
        fig.update_xaxes(title_text = "particles",row = 1, col = 1)
        fig.update_yaxes(title_text = "RMSE",row = 1, col = 1)
        fig.update_xaxes(title_text = "particles",range = [0,self.df_position.index.max()],row =1, col = 2)
        fig.update_yaxes(title_text = "LC",range = [0,1], row =1, col = 2)

        ### F - Phi plot ###
        
        fig.add_trace(go.Scatter(x=self.df_performance[Phi], y=self.df_performance[F],
                                mode='lines',
                                line = dict(color = "lightgray"),
                                name='Simulated'),row =2, col =1)
        fig.add_trace(go.Scatter(x = self.df_best_performance[Phi], # make misfit value a criterion that can be changed
                                y = self.df_best_performance[F],
                                mode = "lines",
                                line = dict(color = "magenta"),
                                text = "nothing yet",
                                name = "best simulations"),row =2, col =1)
        fig.add_trace(go.Scatter(x = self.FD_targets[self.data_to_process[0]]["Phi_interpolated"], y = self.FD_targets[self.data_to_process[0]]["F_interpolated"],
                                mode = "lines",
                                line = dict(color = "red", width = 3),
                                name = "target"),row =2, col =1)
        fig.add_trace(go.Scatter(x = [0,1], y = [0,1],
                                mode = "lines",
                                line = dict(color = "black", width = 3),
                                name = "homogeneous"),row =2, col =1)
        fig.update_xaxes(title_text = "Phi", range = [0,1],row =2, col =1)
        fig.update_yaxes(title_text = "F",range = [0,1], row =2, col = 1)

        ### Sweep efficiency plot ###
        
        for i in range (0,self.df_performance.iteration.max()):
            iteration = i 
            for j in range(0,self.df_performance.particle_no.max()):
                particle_no = j
                EV = self.df_performance[(self.df_performance.iteration == iteration) & (self.df_performance.particle_no == particle_no)][EV_index]
                tD = self.df_performance[(self.df_performance.iteration == iteration) & (self.df_performance.particle_no == particle_no)][tD_index]
                fig.add_trace(go.Scatter(x=tD, y=EV,
                                    mode='lines',
                                    line = dict(color = "lightgray"),
                                    text = "nothing yet",
                                    name = "Simulated"),row =2, col =2)

        for i in range (0,self.df_performance.iteration.max()):
            iteration = i 
            for j in range(0,self.df_performance.particle_no.max()):
                particle_no = j
                EV = self.df_best_performance[(self.df_best_performance.iteration == iteration) & (self.df_best_performance.particle_no == particle_no)][EV_index]
                tD = self.df_best_performance[(self.df_best_performance.iteration == iteration) & (self.df_best_performance.particle_no == particle_no)][tD_index]
                fig.add_trace(go.Scatter(x=tD, y=EV,
                                    mode='lines',
                                    line = dict(color = "magenta"),
                                    text = "nothing yet",
                                    name = "best simulations"),row =2, col =2)

        fig.update_xaxes(title_text = "tD", range = [0,1],row =2, col =2)
        fig.update_yaxes(title_text = "EV",range = [0,1], row =2, col = 2)
        fig.update_layout(title='Performance Evaluation - Simulation run {}'.format(self.data_to_process[0]),
                        autosize = False,
                        width = 1000,
                        height = 1000,
                        showlegend = False)

        fig.show()

    def plot_tof_best_models(self,shedule_no = 0):
        "plot the median tof for best models to see what areas are unswept in most scenarios"
        nx = 200
        ny = 100
        nz = 7
        all_cells_median = []
        tof = "tof_" + str(shedule_no)
        df_best_tof = self.best_tof_dict[tof]
        for i in range(df_best_tof.shape[1]-2):
            # single cell in all models, from seconds to years    
            cell = np.round(np.array(df_best_tof[i]).reshape(-1)/60/60/24/365.25)
            cell_binned = np.digitize(cell,bins=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
            # calculate entropy based upon clusters
            cell_median = np.median(cell_binned)
            all_cells_median.append(cell_median)
            
        # plot the whole thing
        values = np.array(all_cells_median).reshape(nx,ny,nz)
        # Create the spatial reference
        grid = pv.UniformGrid()
        # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
        grid.dimensions = np.array(values.shape) + 1
        # Edit the spatial reference
        grid.origin = (1, 1, 1)  # The bottom left corner of the data set
        grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis
        # Add the data values to the cell data
        grid.cell_arrays["Median tof in years"] =values.flatten()# np.log10(tof)# np.log10(tof)# values.flatten(order="C")  # Flatten the array! C F A K
        boring_cmap = plt.cm.get_cmap("viridis")
        grid.plot(show_edges=False,cmap = boring_cmap)

    def plot_tof_entropy(self,shedule_no = 0):

        """ plot distribution of entropy based on grid for the best models """
        nx = 200
        ny = 100
        nz = 7
        all_cells_entropy = []
        all_cell_cluster_id = []
        tof = "tof_" + str(shedule_no)
        df_best_tof = self.best_tof_dict[tof]
        for i in range(df_best_tof.shape[1]-2):
            # single cell in all models, from seconds to years    
            cell = np.round(np.array(df_best_tof[i]).reshape(-1)/60/60/24/365.25)
            cell_binned = np.digitize(cell,bins=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
            # calculate entropy based upon clusters
            cell_entropy = np.array(ent.shannon_entropy(cell_binned))
            all_cells_entropy.append(cell_entropy)
            
        # plot the whole thing    
        values = np.array(all_cells_entropy).reshape(nx,ny,nz)
        # Create the spatial reference
        grid = pv.UniformGrid()
        # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
        grid.dimensions = np.array(values.shape) + 1
        # Edit the spatial reference
        grid.origin = (1, 1, 1)  # The bottom left corner of the data set
        grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis
        # Add the data values to the cell data
        grid.cell_arrays["Entropy"] =values.flatten()# np.log10(tof)# np.log10(tof)# values.flatten(order="C")  # Flatten the array! C F A K
        boring_cmap = plt.cm.get_cmap("viridis")
        grid.plot(show_edges=False,cmap = boring_cmap)

    def plot_best_model(self,random_model = True,model_id = 0,property = "PORO",shedule_no = 0):
        "visualize the properties of either a radom best model or a specific best model"

        if random_model == True:
            model_id = int(np.random.choice(self.df_best_position.index.values,1))   
        print("Plotting Model {}".format(model_id))
        #get filepath and laod grid

        if property == "tof":
            tof_index = "tof_" + str(shedule_no)
            df_best_tof = self.best_tof_dict[tof_index]

            tof = df_best_tof.loc[model_id,:].drop(["iteration","particle_no"])

            # plot the whole thing
            nx = 200
            ny = 100
            nz = 7
            #tof to years and then binned
            values = np.array(tof).reshape(nx,ny,nz)/60/60/24/365.25
            values = np.digitize(values,bins=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

            # Create the spatial reference
            grid = pv.UniformGrid()

            # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
            grid.dimensions = np.array(values.shape) + 1

            # Edit the spatial reference
            grid.origin = (1, 1, 1)  # The bottom left corner of the data set
            grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

            # Add the data values to the cell data
            grid.cell_arrays["tof"] =values.flatten()# np.log10(tof)# np.log10(tof)# values.flatten(order="C")  # Flatten the array! C F A K

            boring_cmap = plt.cm.get_cmap("viridis")
            grid.plot(show_edges=False,cmap = boring_cmap)

        else:   
            geomodel_path = str(self.setup["base_path"] / "../../Output/"/ self.data_to_process[0] / "all_models/INCLUDE/GRID.GRDECL")
            property_path = str(self.setup["base_path"] /  "../../Output/"/ self.data_to_process[0] / "all_models/INCLUDE" / property / "M{}.GRDECL".format(model_id))
            Model = GeologyModel(filename = geomodel_path)
            TempData = Model.LoadCellData(varname=property,filename=property_path)

            Model.GRDECL2VTK()
            Model.Write2VTU()
            Model.Write2VTP()

            # visulalize
            mesh = pv.read('GRDECL_file_reader\Results\GRID.vtp')
            mesh.plot(scalars = property,show_edges=False, notebook=False)

    def clustering_tof_or_PSO(self,n_neighbors = 30,min_dist = 0,n_components = 30, min_cluster_size = 10,
                              min_samples = 1,allow_single_cluster = True,cluster_parameter = "tof", shedule_no = 0):
        """ Clustering with the help of UMAP (dimension reduction) and HDBSCAN (density based hirachical clustering algo)
            Args: 
             control of UMAP:
                            - n_neighbors = This parameter controls how UMAP balances local versus global structure in the data.
                                            It does this by constraining the size of the local neighborhood UMAP will look at when attempting to learn the manifold structure of the data.
                            - min_dist = closeness of similar data points. 0 good for clustering
                            - n_components = reduce to how many dimensions
            control of HDBSCAN:
                            - min_cluster size = min size of cluster.
                            - min_samples = the smaller, the less points get left out of clustering. --> less conservative.
            cluster_parameter = can either cluster on the PSO_parameters or on  time of flight (tof)
            misfit_tolerance = which models are considered good matches and will be used for clustering
        """

        # turn off the settingwithcopy warning of pandas
        pd.set_option('mode.chained_assignment', None)

        # model parameters that generate lowest misfit
        # particle_parameters used for clustering
        if cluster_parameter == "PSO_parameters":
            # LC = "LC_" + str(shedule_no)
            columns = self.setup_all[self.data_to_process[0]]["columns"]
            df_best_for_clustering = self.df_best_position[columns]
            # df_best_for_clustering["LC"] = self.df_best_position[LC]

            # Create UMAP reducer
            reducer    = umap.UMAP(n_neighbors=n_neighbors,min_dist = min_dist, n_components =n_components)
            embeddings = reducer.fit_transform(df_best_for_clustering)


            # Create HDBSCAN clusters
            hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
            #                      cluster_selection_epsilon= 0.5,
                                min_samples = min_samples,
                                allow_single_cluster= allow_single_cluster
                                )
            scoreTitles = hdb.fit(embeddings)

            self.df_best_position["cluster_PSO_parameters_prob"] = scoreTitles.probabilities_
            self.df_best_position["cluster_PSO_parameters"] = scoreTitles.labels_
            self.df_best_position["cluster_PSO_parameters_x"] =  embeddings[:,0]
            self.df_best_position["cluster_PSO_parameters_y"] = embeddings[:,1]

            fig = go.Figure(data=go.Scatter(x = embeddings[:,0],
                                            y = embeddings[:,1],

                                            mode='markers',
                                            text = self.df_best_position.index,
                                            marker=dict(
                                                size=16,
                                                color=self.df_best_position.cluster_PSO_parameters, #set color equal to a variable
                                                colorscale= "deep",#'Viridis', # one of plotly colorscales
                                                showscale=True,
                                                colorbar=dict(title="Clusters")
                                                )
                                            ))
            fig.update_layout(title='Clustering of {} best models - Number of clusters found: {} - Unclustered models: {}'.format(self.df_best_position.shape[0],self.df_best_position.cluster_PSO_parameters.max()+1,abs(self.df_best_position.cluster_PSO_parameters[self.df_best_position.cluster_PSO_parameters == -1].sum())))
            fig.show()

        elif cluster_parameter == "tof":
            tof = "tof_" + str(shedule_no)
            df_best_tof = self.best_tof_dict[tof]            
            df_best_for_clustering = df_best_tof.drop(columns = ["particle_no","iteration"])

            #convert to years and put in bins.
            df_best_for_clustering = df_best_for_clustering/60/60/24/365.25
            df_best_for_clustering = np.digitize(df_best_for_clustering,bins=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])


            # Create UMAP reducer
            reducer    = umap.UMAP(n_neighbors=n_neighbors,min_dist = min_dist, n_components =n_components)
            embeddings = reducer.fit_transform(df_best_for_clustering)


            # Create HDBSCAN clusters
            hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
            #                      cluster_selection_epsilon= 0.5,
                                min_samples = min_samples,
                                allow_single_cluster= allow_single_cluster
                                )
            scoreTitles = hdb.fit(embeddings)

            self.df_best_position["cluster_tof_prob"] = scoreTitles.probabilities_
            self.df_best_position["cluster_tof"] = scoreTitles.labels_
            self.df_best_position["cluster_tof_x"] =  embeddings[:,0]
            self.df_best_position["cluster_tof_y"] = embeddings[:,1]

            fig = go.Figure(data=go.Scatter(x = embeddings[:,0],
                                            y = embeddings[:,1],

                                            mode='markers',
                                            text = self.df_best_position.index,
                                            marker=dict(
                                                size=16,
                                                color=self.df_best_position.cluster_tof, #set color equal to a variable
                                                colorscale= "deep",#'Viridis', # one of plotly colorscales
                                                showscale=True,
                                                colorbar=dict(title="Clusters")
                                                )
                                            ))
            fig.update_layout(title='Clustering of {} best models - Number of clusters found: {} - Unclustered models: {}'.format(self.df_best_position.shape[0],self.df_best_position.cluster_tof.max()+1,abs(self.df_best_position.cluster_tof[self.df_best_position.cluster_tof == -1].sum())))
            fig.show()
    
    def clustering_sweep_efficiency_or_F_Phi(self, n_neighbors = 30, min_dist = 0, n_components = 30,
                                    min_cluster_size = 5, min_samples = 1, allow_single_cluster = True,
                                    cluster_parameter = "sweep",shedule_no = 0):
        """ use a combination of UMAP (dimension reduction) and HDBSCAN (hirachical density based clustering) to find different "sweep" / "F_Phi" patterns of models
            control of UMAP:
                            - n_neighbors = This parameter controls how UMAP balances local versus global structure in the data.
                                            It does this by constraining the size of the local neighborhood UMAP will look at when attempting to learn the manifold structure of the data.
                            - min_dist = closeness of similar data points. 0 good for clustering
                            - n_components = reduce to how many dimensions
            control of HDBSCAN:
                            - min_cluster size = min size of cluster.
                            - min_samples = the smaller, the less points get left out of clustering. --> less conservative.
        """
        # turn off the settingwithcopy warning of pandas
        pd.set_option('mode.chained_assignment', None)

        if cluster_parameter == "sweep":       
            # filter out tD and EV
            EV_index = "EV_" + str(shedule_no)
            tD_index = "tD_" + str(shedule_no)
            iteration = self.df_best_position.iteration.tolist()
            particle_no =  self.df_best_position.particle_no.tolist()
            EV_all = pd.DataFrame()
            tD_all = pd.DataFrame()
            for i in range(self.df_best_position.shape[0]):

                EV = self.df_performance[(self.df_performance.iteration == iteration[i]) & (self.df_performance.particle_no == particle_no[i])][EV_index]
                tD = self.df_performance[(self.df_performance.iteration == iteration[i]) & (self.df_performance.particle_no == particle_no[i])][tD_index]
                EV.reset_index(drop=True, inplace=True)
                tD.reset_index(drop=True, inplace=True)
                EV_all = pd.concat([EV_all,EV],ignore_index=True,axis = 1)
                tD_all = pd.concat([tD_all,tD],ignore_index=True,axis = 1)
            
            EV_tD = tD_all.append(EV_all,ignore_index = True)
            # reduce points used to every 10th point
            EV_tD = EV_tD.iloc[::10]
            EV_tD = EV_tD.T
            self.df_best_sweep_efficiency = EV_tD

            # # Create UMAP reducer
            reducer    = umap.UMAP(n_neighbors=n_neighbors,min_dist = min_dist, n_components = n_components)
            embeddings = reducer.fit_transform(self.df_best_sweep_efficiency)

            # Create HDBSCAN clusters
            hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
            #                       cluster_selection_epsilon= 0.5,
                                min_samples = min_samples,
                                allow_single_cluster= allow_single_cluster
                                )
            scoreTitles = hdb.fit(embeddings)

            self.df_best_sweep_efficiency["cluster_sweep_prob"] = scoreTitles.probabilities_
            self.df_best_sweep_efficiency["cluster_sweep"] = scoreTitles.labels_
            self.df_best_sweep_efficiency["cluster_sweep_x"] = embeddings[:,0]
            self.df_best_sweep_efficiency["cluster_sweep_y"] = embeddings[:,1]
            self.df_best_performance["cluster_sweep"] = np.nan
            self.df_best_position["cluster_sweep"] = np.nan
            self.df_best_position["cluster_sweep_x"] = embeddings[:,0]
            self.df_best_position["cluster_sweep_y"] = embeddings[:,1]

            # generate colors for sweep eff. cluster plot
            #deep start & end hex color
            c0 = "#FDFDCC" # beige
            c1 = "#271A2C" # dark blue

            # list of "N" (n_clusters) colors between "start_color" and "end_color"
            colorscale = [x.hex for x in list(Color(c0).range_to(Color(c1), self.df_best_sweep_efficiency["cluster_sweep"].max()+2))]

            # iteration = self.df_best_position.iteration.tolist()
            # particle_no =  self.df_best_position.particle_no.tolist()
            cluster_sweep = self.df_best_sweep_efficiency.cluster_sweep.tolist()
            for i in range(self.df_best_position.shape[0]):
                self.df_best_performance.cluster_sweep[(self.df_best_performance.iteration == iteration[i]) & (self.df_best_performance.particle_no == particle_no[i])] = cluster_sweep[i]
                self.df_best_position.cluster_sweep[(self.df_best_position.iteration == iteration[i]) & (self.df_best_position.particle_no == particle_no[i])] = cluster_sweep[i]

            fig = make_subplots(rows = 1, cols = 1)

            for i in range(self.df_best_position.shape[0]):
                EV = self.df_best_performance[(self.df_best_performance.iteration == iteration[i]) & (self.df_best_performance.particle_no == particle_no[i])][EV_index]
                tD = self.df_best_performance[(self.df_best_performance.iteration == iteration[i]) & (self.df_best_performance.particle_no == particle_no[i])][tD_index]
                cluster_sweep = int(self.df_best_performance[(self.df_best_performance.iteration == iteration[i]) & (self.df_best_performance.particle_no == particle_no[i])].cluster_sweep.unique())
                fig.add_trace(go.Scatter(x=tD, y=EV,
                                    mode='lines',
                                    line = dict(color = colorscale[cluster_sweep]),
                                    text =  cluster_sweep))
                
            fig.update_xaxes(title_text = "tD", range = [0,2],row =1, col =1)
            fig.update_yaxes(title_text = "Ev",range = [0,1], row =1, col = 1)
            fig.update_layout(title="Sweep Efficiency Clustered - Number of clusters found: {} - Unclustered models: {}".format(self.df_best_sweep_efficiency.cluster_sweep.max()+1,abs(self.df_best_sweep_efficiency.cluster_sweep[self.df_best_sweep_efficiency.cluster_sweep == -1].sum())),
                            autosize = False,
                            width = 1000,
                            height = 1000,
                            showlegend = False)
            fig.show()

        elif cluster_parameter == "F_Phi":

            # filter out tD and EV
            F_index = "F_" + str(shedule_no)
            Phi_index = "Phi_" + str(shedule_no)
            iteration = self.df_best_position.iteration.tolist()
            particle_no =  self.df_best_position.particle_no.tolist()
            F_all = pd.DataFrame()
            Phi_all = pd.DataFrame()
            for i in range(self.df_best_position.shape[0]):

                F = self.df_performance[(self.df_performance.iteration == iteration[i]) & (self.df_performance.particle_no == particle_no[i])][F_index]
                Phi = self.df_performance[(self.df_performance.iteration == iteration[i]) & (self.df_performance.particle_no == particle_no[i])][Phi_index]
                F.reset_index(drop=True, inplace=True)
                Phi.reset_index(drop=True, inplace=True)
                F_all = pd.concat([F_all,F],ignore_index=True,axis = 1)
                Phi_all = pd.concat([Phi_all,Phi],ignore_index=True,axis = 1)
            
            F_Phi = F_all.append(Phi_all,ignore_index = True)
            # reduce points used to every 10th point
            F_Phi = F_Phi.iloc[::10]
            F_Phi = F_Phi.T
            self.df_best_F_Phi = F_Phi

            # # Create UMAP reducer
            reducer    = umap.UMAP(n_neighbors=n_neighbors,min_dist = min_dist, n_components = n_components)
            embeddings = reducer.fit_transform(self.df_best_F_Phi)

            # Create HDBSCAN clusters
            hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
            #                       cluster_selection_epsilon= 0.5,
                                min_samples = min_samples,
                                allow_single_cluster= allow_single_cluster
                                )
            scoreTitles = hdb.fit(embeddings)

            self.df_best_F_Phi["cluster_F_Phi"] = scoreTitles.labels_
            self.df_best_F_Phi["cluster_F_Phi_x"] = embeddings[:,0]
            self.df_best_F_Phi["cluster_F_Phi_y"] = embeddings[:,1]
            self.df_best_performance["cluster_F_Phi"] = np.nan
            self.df_best_position["cluster_F_Phi"] = np.nan
            self.df_best_position["cluster_F_Phi_x"] = embeddings[:,0]
            self.df_best_position["cluster_F_Phi_y"] = embeddings[:,1]

            # generate colors for sweep eff. cluster plot
            #deep start & end hex color
            c0 = "#FDFDCC" # beige
            c1 = "#271A2C" # dark blue

            # list of "N" (n_clusters) colors between "start_color" and "end_color"
            colorscale = [x.hex for x in list(Color(c0).range_to(Color(c1), self.df_best_F_Phi["cluster_F_Phi"].max()+2))]

            cluster_F_Phi = self.df_best_F_Phi.cluster_F_Phi.tolist()
            for i in range(self.df_best_position.shape[0]):
                self.df_best_performance.cluster_F_Phi[(self.df_best_performance.iteration == iteration[i]) & (self.df_best_performance.particle_no == particle_no[i])] = cluster_F_Phi[i]
                self.df_best_position.cluster_F_Phi[(self.df_best_position.iteration == iteration[i]) & (self.df_best_position.particle_no == particle_no[i])] = cluster_F_Phi[i]

            fig = make_subplots(rows = 1, cols = 1)

            for i in range(self.df_best_position.shape[0]):
                F = self.df_best_performance[(self.df_best_performance.iteration == iteration[i]) & (self.df_best_performance.particle_no == particle_no[i])][F_index]
                Phi = self.df_best_performance[(self.df_best_performance.iteration == iteration[i]) & (self.df_best_performance.particle_no == particle_no[i])][Phi_index]
                cluster_F_Phi = int(self.df_best_performance[(self.df_best_performance.iteration == iteration[i]) & (self.df_best_performance.particle_no == particle_no[i])].cluster_F_Phi.unique())
                fig.add_trace(go.Scatter(x=Phi, y=F,
                                    mode='lines',
                                    line = dict(color = colorscale[cluster_F_Phi]),
                                    text =  cluster_F_Phi))
                
            fig.update_xaxes(title_text = "Phi", range = [0,1],row =1, col =1)
            fig.update_yaxes(title_text = "F",range = [0,1], row =1, col = 1)
            fig.update_layout(title="F - Phi Clustered - Number of clusters found: {} - Unclustered models: {}".format(self.df_best_F_Phi.cluster_F_Phi.max()+1,abs(self.df_best_F_Phi.cluster_F_Phi[self.df_best_F_Phi.cluster_F_Phi == -1].sum())),
                            autosize = False,
                            width = 1000,
                            height = 1000,
                            showlegend = False)
            fig.show()

    def cluster_model_selection(self,cluster_parameter = "tof",manual_pick = False,include_unclustered = False,n_reservoir_models = 10,manual_model_id_list = []):
        """ how many models do we want to select from clusters. eitehr from "tof", "PSO_parameter", or "sweep" or "F_Phi"
            randomly sample from each cluster, bigger clusters will get bigger representation
            also option to manually put in the id of models that we want to select
            manual_model_id_list = [] thats how you can select potential unclusterd models, too
        """
        cluster = "cluster_{}".format(cluster_parameter)
        if manual_pick == False:
            # how many modesl to select from (only clutered ones)            
            n_best_models_total= self.df_best_position[self.df_best_position[cluster] != -1].shape[0]
            
            # how many models to select
            if n_reservoir_models > n_best_models_total:
                n_reservoir_models = n_best_models_total

            #how many models in each cluster
            model_cluster_dist = Counter(self.df_best_position[self.df_best_position[cluster] != -1][cluster])
            models_per_cluster = []
            #depending on cluster size, determine how many models to pick per cluster
            for i in range(len(model_cluster_dist)):
                models_per_cluster.append(np.round(model_cluster_dist[i]/n_best_models_total*n_reservoir_models))
            # check and correct if too many/too few models selected
            
            if np.sum(models_per_cluster) == n_reservoir_models:
                pass
            elif np.sum(models_per_cluster) > n_reservoir_models:
                while np.sum(models_per_cluster) != n_reservoir_models:
                    models_per_cluster[models_per_cluster.index(max(models_per_cluster))] =  models_per_cluster[models_per_cluster.index(max(models_per_cluster))] -1
            elif np.sum(models_per_cluster) < n_reservoir_models:
                while np.sum(models_per_cluster) != n_reservoir_models:
                    models_per_cluster[models_per_cluster.index(min(models_per_cluster))] =  models_per_cluster[models_per_cluster.index(min(models_per_cluster))] +1

            # randomly sample n models according to models_per_cluster from each clusters
            best_model_sampler = self.df_best_position[self.df_best_position[cluster] != -1].copy()
            best_model_sampler["index_1"] = best_model_sampler.index
            column_names = best_model_sampler.columns.values.tolist()
            self.df_best_models_to_save = pd.DataFrame(columns = column_names )

            for i in range(len(models_per_cluster)):
                for j in range(int(models_per_cluster[i])):
                    sample_from_cluster = best_model_sampler[(best_model_sampler[cluster] == i)].sample()
                    #drop the model that has just been sampled
                    best_model_sampler.drop(sample_from_cluster.index_1,inplace = True)
                    self.df_best_models_to_save = pd.concat([self.df_best_models_to_save,sample_from_cluster])
            
            if include_unclustered == True:
             # get models that arent clustered and append them
                if -1 is not self.df_best_position[cluster]:
                    self.df_best_models_to_save = self.df_best_models_to_save.append(self.df_best_position[self.df_best_position[cluster] == -1])
        
        elif manual_pick == True:
        # feed the manual_pick list and select models acorndignly
            best_model_sampler = self.df_best_position.copy()
            best_model_sampler["index_1"] = best_model_sampler.index
            column_names = best_model_sampler.columns.values.tolist()
            self.df_best_models_to_save = pd.DataFrame(columns = column_names )

            for i in range(manual_model_id_list):
                sample = best_model_sampler[(best_model_sampler.index_1 == manual_model_id_list[i])]
                self.df_best_models_to_save = pd.concat([self.df_best_models_to_save,sample])

        # show selected models with all models
        fig = make_subplots(rows = 1, cols = 1)

        cluster_x = "cluster_{}_x".format(cluster_parameter)
        cluster_y = "cluster_{}_y".format(cluster_parameter)


        fig.add_trace(go.Scatter(x = self.df_best_position[cluster_x],
                                 y = self.df_best_position[cluster_y],
                                mode='markers',
                                text = self.df_best_position.index,
                                marker=dict(
                                            size=16,
                                            color=self.df_best_position[cluster], #set color equal to a variable
                                            colorscale= "deep",#'Viridis', # one of plotly colorscales
                                            showscale=True,
                                            colorbar=dict(title="Clusters")
                                            )
                                ))
        fig.add_trace(go.Scatter(x = self.df_best_models_to_save[cluster_x],
                                 y = self.df_best_models_to_save[cluster_y],
                                 mode='markers',
                                 text = self.df_best_position.index,
                                 marker = dict(
                                                size = 14,
                                                color = 0,
                                              )
                                ))
                        
        fig.update_layout(showlegend=False)
        fig.update_layout(title='Clustering of {} best models - Number of clusters found: {} - Unclustered models: {}'.format(self.df_best_position.shape[0],self.df_best_position[cluster].max()+1,abs(self.df_best_position[cluster][self.df_best_position[cluster] == -1].sum())))
        fig.show()

    def save_best_clustered_models(self,shedule_no = 0):
        """ save the best,previously selected reservoir models that are ready for full reservoir simulations
            also generate 2 csv files with df best_position of selected and all best models. """        

        # save best models
        n_datasets = len(self.data_to_process)

        for i in range(self.n_datasets):

            # open df_position to figure out which models performed best
            path = str(self.setup["base_path"] / "../../Output/") + "/" + self.data_to_process[i] + "/"

            n_best_models_selected = len(self.df_best_models_to_save)
            best_models_selected_index = self.df_best_models_to_save.index.tolist()

            #path to all models 
            all_path = path + "all_models/"
            data_all_path = all_path + "DATA/"
            include_all_path = all_path + "INCLUDE/"
            permx_all_path = include_all_path + "PERMX/"
            permy_all_path = include_all_path + "PERMY/"
            permz_all_path = include_all_path + "PERMZ/"
            poro_all_path = include_all_path + "PORO/"

            #path to best models 
            destination_best_path = path + "best_models/"
            data_best_path = destination_best_path + "DATA/"
            include_best_path = destination_best_path + "INCLUDE/"
            permx_best_path = include_best_path + "PERMX/"
            permy_best_path = include_best_path + "PERMY/"
            permz_best_path = include_best_path + "PERMZ/"
            poro_best_path = include_best_path + "PORO/"

            if not os.path.exists(destination_best_path):
                # make folders and subfolders
                os.makedirs(destination_best_path)
                os.makedirs(data_best_path)
                os.makedirs(include_best_path)
                os.makedirs(permx_best_path)
                os.makedirs(permy_best_path)
                os.makedirs(permz_best_path)
                os.makedirs(poro_best_path)

                # save best position df as csv. the selected ones and all best models
                best_position_path = destination_best_path + "best_models_selected.csv"
                best_position_path_2 = destination_best_path + "best_models.csv"
                self.df_best_models_to_save.to_csv(best_position_path,index=False)
                self.df_best_position.to_csv(best_position_path_2,index = False)

                #cop and paste generic files into Data
                DP_pvt_all_path = include_all_path + "DP_pvt.INC"
                GRID_all_path = include_all_path + "GRID.GRDECL"
                ROCK_RELPERMS_all_path = include_all_path + "ROCK_RELPERMS.INC"
                SCHEDULE_all_path = include_all_path + "5_spot.INC"
                SOLUTION_all_path = include_all_path + "SOLUTION.INC"
                SUMMARY_all_path = include_all_path + "SUMMARY.INC"

                DP_pvt_best_path = include_best_path + "DP_pvt.INC"
                GRID_best_path = include_best_path + "GRID.GRDECL"
                ROCK_RELPERMS_best_path = include_best_path + "ROCK_RELPERMS.INC"
                SCHEDULE_best_path = include_best_path + "5_spot.INC"
                SOLUTION_best_path = include_best_path + "SOLUTION.INC"
                SUMMARY_best_path = include_best_path + "SUMMARY.INC"

                shutil.copy(DP_pvt_all_path,DP_pvt_best_path)
                shutil.copy(GRID_all_path,GRID_best_path)
                shutil.copy(ROCK_RELPERMS_all_path,ROCK_RELPERMS_best_path)
                shutil.copy(SCHEDULE_all_path,SCHEDULE_best_path)
                shutil.copy(SOLUTION_all_path,SOLUTION_best_path)
                shutil.copy(SUMMARY_all_path,SUMMARY_best_path)

            if os.path.exists(destination_best_path):
                #copy and paste best models 
                for i in range(n_best_models_selected):
                    
                    data_file_all_path = data_all_path + "M{}_{}.DATA".format(best_models_selected_index[i],shedule_no)  
                    permx_file_all_path = permx_all_path + "M{}.GRDECL".format(best_models_selected_index[i])  
                    permy_file_all_path = permy_all_path + "M{}.GRDECL".format(best_models_selected_index[i])  
                    permz_file_all_path = permz_all_path + "M{}.GRDECL".format(best_models_selected_index[i])  
                    poro_file_all_path = poro_all_path + "M{}.GRDECL".format(best_models_selected_index[i])

                    data_file_best_path = data_best_path + "M{}_{}.DATA".format(best_models_selected_index[i],shedule_no)  
                    permx_file_best_path = permx_best_path + "M{}.GRDECL".format(best_models_selected_index[i])  
                    permy_file_best_path = permy_best_path + "M{}.GRDECL".format(best_models_selected_index[i])  
                    permz_file_best_path = permz_best_path + "M{}.GRDECL".format(best_models_selected_index[i])  
                    poro_file_best_path = poro_best_path + "M{}.GRDECL".format(best_models_selected_index[i]) 

                    shutil.copy(data_file_all_path,data_file_best_path)
                    shutil.copy(permx_file_all_path,permx_file_best_path)
                    shutil.copy(permy_file_all_path,permy_file_best_path)
                    shutil.copy(permz_file_all_path,permz_file_best_path)
                    shutil.copy(poro_file_all_path,poro_file_best_path)

                    # save best position df as csv. the selected ones and all best models
                    best_position_path = destination_best_path + "best_models_selected.csv"
                    best_position_path_2 = destination_best_path + "best_models.csv"
                    # best_tof_path = destination_best_path + "best_models_tof.csv"

                    self.df_best_models_to_save.to_csv(best_position_path,index=False)
                    self.df_best_position.to_csv(best_position_path_2,index = False)
                    # self.df_best_tof.to_csv(best_tof_path,index = False)

    