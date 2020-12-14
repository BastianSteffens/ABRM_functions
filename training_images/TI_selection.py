import numpy as np
import pandas as pd
import bz2
import _pickle as cPickle
import os
import pathlib
import pyvista as pv
from GRDECL_file_reader.GRDECL2VTK import *
from colour import Color
import umap
import hdbscan
from pyentrp import entropy as ent
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import shutil


class TI_selection():
    """ pick Training images wiht help of clustering that should be taken forward """

    def __init__(self,dataset,TI_name= "TI_crest",nx = 60,ny = 60,nz = 1,n_shedules = 1,n_TI = 1200):
    
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.TI_name = TI_name
        self.n_shedules = n_shedules
        self.n_TI = n_TI
        self.base_path = pathlib.Path(__file__).parent  
        path = self.base_path / "../../Output/training_images/{}/".format(self.TI_name)
        self.dataset = dataset
        TI_props = "TI_properties.csv"
        TI_tof = "tof_all_TI.pbz2"
        setup = "setup.pickle"
        TI_performance = "all_TI_performance.pbz2"
        
        folder_path = path / self.dataset
        file_path_TI_props = folder_path / TI_props
        file_path_TI_tof = folder_path / TI_tof
        file_path_setup = folder_path / setup
        file_path_TI_performance = folder_path / TI_performance

        # load data
        self.df_TI_props = pd.read_csv(file_path_TI_props)
        self.columns = self.df_TI_props.columns.values.tolist()
        self.columns.remove("TI_no")

        # load pickle file
        data = bz2.BZ2File(file_path_setup)
        self.setup = cPickle.load(data)

        #load compressed pickle file
        data = bz2.BZ2File(file_path_TI_tof,"rb")
        self.df_tof_raw = cPickle.load(data)

        
        #load compressed pickle file
        data = bz2.BZ2File(file_path_TI_performance,"rb")
        self.df_TI_performance = cPickle.load(data)
        
        #convert to readable format for ploting
        # self.df_tof = pd.DataFrame(columns = np.arange(int(self.nx)*int(self.ny)*int(self.nz)*self.n_shedules))
   
        TI_no = self.df_TI_props.TI_no.tolist()
        tof_all = pd.DataFrame()
        self.df_tof = pd.DataFrame()
        if self.n_shedules == 1:
            for i in range(self.df_TI_props.shape[0]):
                tof = self.df_tof_raw[(self.df_tof_raw.TI_no == TI_no[i])].tof
                tof.reset_index(drop=True, inplace=True)
                tof_all = tof_all.append(tof,ignore_index = True)
            self.df_tof = tof_all
            self.df_tof["TI_no"] = TI_no
            self.df_tof.set_index(self.df_TI_props.index.values,inplace = True)
        
        else:
            # TI_no_extended = list()
            for shedule_no in range(self.n_shedules):
                tof_all = pd.DataFrame()
                tof_name = "tof_" + str(shedule_no)
                # TI_no_extended.extend(TI_no)

                for i in range(self.df_TI_props.shape[0]):

                    tof = self.df_tof_raw[(self.df_tof_raw.TI_no == TI_no[i])][tof_name]
                    tof.reset_index(drop=True, inplace=True)
                    tof_all = tof_all.append(tof,ignore_index = True)

                tof_all = tof_all.add_suffix("_" + str(shedule_no))
                self.df_tof = pd.concat([self.df_tof,tof_all],axis = 1)
            self.df_tof["TI_no"] = TI_no


    def plot_TI_performance(self):
        """ Create plots showing the Flow Diagnostic Performance of the PSO over n iterations
            Args: filter dataset to data that are fulfill misfit tolerance """
        
        if self.n_shedules == 1:
            # Create traces
            
            fig = make_subplots(rows = 1, cols = 3,
                            subplot_titles = ("LC plot","F - Phi Graph","Sweep Efficieny Graph"))

            ### LC plot ###
            
            fig.add_trace(go.Scatter(x = self.df_TI_performance.index, y=self.df_TI_performance.LC,
                                    mode='markers',
                                    line = dict(color = "lightgray"),
                                    name='Simulated'),row =1, col =1)

            fig.update_xaxes(title_text = "TI_no",range = [0,self.df_TI_performance.TI_no.max()],row =1, col = 1)
            fig.update_yaxes(title_text = "LC",range = [0,1], row =1, col = 1)

            ### F - Phi plot ###
            
            fig.add_trace(go.Scatter(x=self.df_TI_performance.Phi, y=self.df_TI_performance.F,
                                    mode='lines',
                                    line = dict(color = "lightgray"),
                                    name='Simulated'),row =1, col =2)
        
            fig.add_trace(go.Scatter(x = [0,1], y = [0,1],
                                    mode = "lines",
                                    line = dict(color = "black", width = 3),
                                    name = "homogeneous"),row =1, col =2)
            fig.update_xaxes(title_text = "Phi", range = [0,1],row =1, col =2)
            fig.update_yaxes(title_text = "F",range = [0,1], row =1, col = 2)

            ### Sweep efficiency plot ###
            
            for TI_no in range (0,self.df_TI_performance.TI_no.max()):

                EV = self.df_TI_performance[(self.df_TI_performance.TI_no == TI_no)].EV
                tD = self.df_TI_performance[(self.df_TI_performance.TI_no == TI_no)].tD
                fig.add_trace(go.Scatter(x=tD, y=EV,
                                    mode='lines',
                                    line = dict(color = "lightgray"),
                                    text = "nothing yet",
                                    name = "Simulated"),row =1, col =3)

            fig.update_xaxes(title_text = "tD", range = [0,1],row =1, col =3)
            fig.update_yaxes(title_text = "Ev",range = [0,1], row =1, col = 3)
            fig.update_layout(title='Performance Evaluation - TI generator -{}'.format(self.dataset),
                            autosize = False,
                            width = 1500,
                            height = 500,
                            showlegend = False)

            fig.show()
        
        else:

            # Create traces
            
            fig = make_subplots(rows = self.n_shedules, cols = 3,
                                subplot_titles = ("LC plot","F - Phi Graph","Sweep Efficieny Graph"))
            for shedule_no in range(self.n_shedules):

                    
                EV = "EV_" + str(shedule_no)
                tD = "tD_" + str(shedule_no)
                F = "F_" + str(shedule_no)
                Phi = "Phi_" + str(shedule_no)
                LC = "LC_" + str(shedule_no)
                tof = "tof_" + str(shedule_no)

                ### LC plot ###
                
                fig.add_trace(go.Scatter(x = self.df_TI_performance.TI_no, y=self.df_TI_performance[LC],
                                        mode='markers',
                                        line = dict(color = "lightgray"),
                                        name='Simulated'),row =shedule_no+1, col =1)

                fig.update_xaxes(title_text = "TI_no",range = [0,self.df_TI_performance.TI_no.max()],row =shedule_no+1, col = 1)
                fig.update_yaxes(title_text = "LC",range = [0,1], row =shedule_no+1, col = 1)

                ### F - Phi plot ###
                
                fig.add_trace(go.Scatter(x=self.df_TI_performance[Phi], y=self.df_TI_performance[F],
                                        mode='lines',
                                        line = dict(color = "lightgray"),
                                        name='Simulated'),row =shedule_no+1, col =2)
            
                fig.add_trace(go.Scatter(x = [0,1], y = [0,1],
                                        mode = "lines",
                                        line = dict(color = "black", width = 3),
                                        name = "homogeneous"),row = shedule_no+1, col =2)
                fig.update_xaxes(title_text = "Phi", range = [0,1],row =shedule_no+1, col =2)
                fig.update_yaxes(title_text = "F",range = [0,1], row =shedule_no+1, col = 2)

                ### Sweep efficiency plot ###
                
                for TI_no in range (0,self.df_TI_performance.TI_no.max()):

                    EV_plt = self.df_TI_performance[(self.df_TI_performance.TI_no == TI_no)][EV]
                    tD_plt = self.df_TI_performance[(self.df_TI_performance.TI_no == TI_no)][tD]
                    fig.add_trace(go.Scatter(x=tD_plt, y=EV_plt,
                                        mode='lines',
                                        line = dict(color = "lightgray"),
                                        text = "nothing yet",
                                        name = "Simulated"),row =shedule_no+1, col =3)

                fig.update_xaxes(title_text = "tD", range = [0,1],row =shedule_no+1, col =3)
                fig.update_yaxes(title_text = "Ev",range = [0,1], row =shedule_no+1, col = 3)
                fig.update_layout(title='Performance Evaluation - TI generator -{}'.format(self.dataset),
                                autosize = False,
                                width = 1500,
                                height = 500*self.n_shedules,
                                showlegend = False)

            fig.show()
        
    def plot_tof_all_models(self,min_tof_bin = 1,max_tof_bin = 20, n_bins = 20,shedule_no = 0):
        "plot the median tof for TI to see what areas are unswept in most scenarios"

        all_cells_median = []
        bins = np.linspace(min_tof_bin,max_tof_bin,n_bins)

        if self.n_shedules == 1:
        
            for i in range(self.df_tof.shape[1]-1):
                # single cell in all models, from seconds to years    
                cell = np.round(np.array(self.df_tof[i]).reshape(-1)/60/60/24/365.25)
                cell_binned = np.digitize(cell,bins=bins)
                for j in range(1,n_bins+1):
                    cell_binned[cell_binned == j] = bins[j-1]
                # calculate entropy based upon clusters
                cell_median = np.median(cell_binned)
                all_cells_median.append(cell_median)
        
        else:
            
            shedule_id = "_" + str(shedule_no)
            col_list = [col for col in self.df_tof.columns if shedule_id in col]
            for i in range(len(col_list)):
                # single cell in all models, from seconds to years    
                cell = np.round(np.array(self.df_tof[col_list[i]]).reshape(-1)/60/60/24/365.25)
                cell_binned = np.digitize(cell,bins=bins)
                for j in range(1,n_bins+1):
                    cell_binned[cell_binned == j] = bins[j-1]
                # calculate entropy based upon clusters
                cell_median = np.median(cell_binned)
                all_cells_median.append(cell_median)
                
        # plot the whole thing
        values = np.array(all_cells_median).reshape(self.nx,self.ny,self.nz)
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

    def plot_tof_entropy(self,min_tof_bin = 1,max_tof_bin = 20, n_bins = 20,shedule_no = 0):
        """ plot distribution of entropy based on grid for all TIs """

        all_cells_entropy = []
        all_cell_cluster_id = []
        bins = np.linspace(min_tof_bin,max_tof_bin,n_bins)

        if self.n_shedules == 1:

            for i in range(self.df_tof.shape[1]-1):
                # single cell in all models, from seconds to years    
                cell = np.round(np.array(self.df_tof[i]).reshape(-1)/60/60/24/365.25)
                cell_binned = np.digitize(cell,bins= bins)
                for j in range(1,n_bins+1):
                    cell_binned[cell_binned == j] = bins[j-1]
                # calculate entropy based upon clusters
                cell_entropy = np.array(ent.shannon_entropy(cell_binned))
                all_cells_entropy.append(cell_entropy)
        
        else:
            
            shedule_id = "_" + str(shedule_no)
            col_list = [col for col in self.df_tof.columns if shedule_id in col]
            for i in range(len(col_list)):
                # single cell in all models, from seconds to years    
                cell = np.round(np.array(self.df_tof[col_list[i]]).reshape(-1)/60/60/24/365.25)
                cell_binned = np.digitize(cell,bins= bins)
                for j in range(1,n_bins+1):
                    cell_binned[cell_binned == j] = bins[j-1]
                # calculate entropy based upon clusters
                cell_entropy = np.array(ent.shannon_entropy(cell_binned))
                all_cells_entropy.append(cell_entropy)
                

        # plot the whole thing    
        values = np.array(all_cells_entropy).reshape(self.nx,self.ny,self.nz)
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

    def plot_best_model(self,random_TI = True,TI_id = 0,property = "PORO",min_tof_bin = 1,max_tof_bin = 20, n_bins = 20,shedule_no = 0):
        "visualize the properties of either a radom Training image model or a specific best model"
        
        if random_TI == True:
            TI_id = int(np.random.choice(self.df_TI_props.index.values,1))   

        print("Plotting Training Image {}".format(TI_id))
        #get filepath and laod grid

        if property == "tof":

            if self.n_shedules == 1:
                tof = self.df_tof.loc[TI_id,:].drop(["TI_no"])
            else:
                shedule_id = "_" + str(shedule_no)
                col_list = [col for col in self.df_tof.columns if shedule_id in col]
                tof = self.df_tof.loc[TI_id,:]
                tof = tof[col_list]

            #tof to years and then binned
            values = np.array(tof)/60/60/24/365.25
            bins = np.linspace(min_tof_bin,max_tof_bin,n_bins)
            values = np.digitize(values,bins= bins)

            for i in range(1,n_bins+1):
                values[values == i] = bins[i-1]

            values = values.reshape(self.nx,self.ny,self.nz)

            # Create the spatial reference
            grid = pv.UniformGrid()

            # Set the grid dimensions: shape + 1 because we want to inject our values on the CELL data
            # grid.dimensions = np.array(values.shape) + 1
            grid.dimensions = np.array(values.shape) + 1

            # Edit the spatial reference
            grid.origin = (1, 1, 1)  # The bottom left corner of the data set
            grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

            # Add the data values to the cell data
            # grid.cell_arrays["tof"] =values.flatten()# np.log10(tof)# np.log10(tof)# values.flatten(order="C")  # Flatten the array! C F A K
            grid.cell_arrays["tof"] =values.flatten()# np.log10(tof)# np.log10(tof)# values.flatten(order="C")  # Flatten the array! C F A K

            boring_cmap = plt.cm.get_cmap("viridis")
            grid.plot(show_edges=False,cmap = boring_cmap)

        else:   
            geomodel_path = str(self.base_path / "../../Output/training_images/{}/".format(self.TI_name) / self.dataset / "all_models/INCLUDE/GRID.GRDECL")
            property_path = str(self.base_path / "../../Output/training_images/{}/".format(self.TI_name) / self.dataset / "all_models/INCLUDE" / property / "TI{}.GRDECL".format(TI_id))

            Model = GeologyModel(filename = geomodel_path)
            TempData = Model.LoadCellData(varname=property,filename=property_path)

            Model.GRDECL2VTK()
            Model.Write2VTU()
            Model.Write2VTP()

            # visulalize
            mesh = pv.read('GRDECL_file_reader\Results\GRID.vtp')
            mesh.plot(scalars = property,show_edges=False, notebook=False)

    def clustering_tof_or_TI_props(self,n_neighbors = 30,min_dist = 0,n_components = 30, min_cluster_size = 10,
                                min_samples = 1,allow_single_cluster = True,cluster_parameter = "tof",min_tof_bin = 1,max_tof_bin = 20, n_bins = 20):
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
            cluster_parameter = can either cluster on the TI_props or on  time of flight (tof)
            misfit_tolerance = which models are considered good matches and will be used for clustering
        """

        # turn off the settingwithcopy warning of pandas
        pd.set_option('mode.chained_assignment', None)

        # model parameters that generate lowest misfit
        # particle_parameters used for clustering
        if cluster_parameter == "TI_props":

            # df_best_for_clustering = self.df_TI_props.drop(["TI_no"], axis = 1)
            df_best_for_clustering = self.df_TI_props[self.columns]

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

            self.df_TI_props["cluster_TI_props_prob"] = scoreTitles.probabilities_
            self.df_TI_props["cluster_TI_props"] = scoreTitles.labels_
            self.df_TI_props["cluster_TI_props_x"] =  embeddings[:,0]
            self.df_TI_props["cluster_TI_props_y"] = embeddings[:,1]

            fig = go.Figure(data=go.Scatter(x = embeddings[:,0],
                                            y = embeddings[:,1],

                                            mode='markers',
                                            text = self.df_TI_props.index,
                                            marker=dict(
                                                size=16,
                                                color=self.df_TI_props.cluster_TI_props_prob, #set color equal to a variable
                                                colorscale= "deep",#'Viridis', # one of plotly colorscales
                                                showscale=True,
                                                colorbar=dict(title="Clusters")
                                                )
                                            ))
            fig.update_layout(title='Clustering of {} Training Images - Number of clusters found: {} - Unclustered models: {}'.format(self.df_TI_props.shape[0],self.df_TI_props.cluster_TI_props_prob.max()+1,abs(self.df_TI_props.cluster_TI_props_prob[self.df_TI_props.cluster_TI_props_prob == -1].sum())))
            fig.show()

        elif cluster_parameter == "tof":
            
            df_best_for_clustering = self.df_tof.drop(columns = ["TI_no"], axis = 1)

            #convert to years and put in bins.
            df_best_for_clustering = df_best_for_clustering/60/60/24/365.25
            bins = np.linspace(min_tof_bin,max_tof_bin,n_bins)
            df_best_for_clustering = np.digitize(df_best_for_clustering,bins=bins)

            for i in range(1,n_bins+1):
                df_best_for_clustering[df_best_for_clustering == i] = bins[i-1]
     

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

            self.df_TI_props["cluster_tof_prob"] = scoreTitles.probabilities_
            self.df_TI_props["cluster_tof"] = scoreTitles.labels_
            self.df_TI_props["cluster_tof_x"] =  embeddings[:,0]
            self.df_TI_props["cluster_tof_y"] = embeddings[:,1]

            fig = go.Figure(data=go.Scatter(x = embeddings[:,0],
                                            y = embeddings[:,1],

                                            mode='markers',
                                            text = self.df_TI_props.index,
                                            marker=dict(
                                                size=16,
                                                color=self.df_TI_props.cluster_tof, #set color equal to a variable
                                                colorscale= "deep",#'Viridis', # one of plotly colorscales
                                                showscale=True,
                                                colorbar=dict(title="Clusters")
                                                )
                                            ))
            fig.update_layout(title='Clustering of {} best models - Number of clusters found: {} - Unclustered models: {}'.format(self.df_TI_props.shape[0],self.df_TI_props.cluster_tof.max()+1,abs(self.df_TI_props.cluster_tof[self.df_TI_props.cluster_tof == -1].sum())))
            fig.show()

    def cluster_TI_selection(self,cluster_parameter = "tof",manual_pick = False,include_unclustered = False,n_TIs = 10,manual_TI_id_list = []):
        """ how many Training images do we want to select from clusters. eitehr from "tof", "TI_props", or "sweep" or "F_Phi"
            randomly sample from each cluster, bigger clusters will get bigger representation
            also option to manually put in the id of models that we want to select
            manual_TI_id_list = [] thats how you can select potential unclusterd Training Images, too
        """
        cluster = "cluster_{}".format(cluster_parameter)
        if manual_pick == False:
            # how many modesl to select from (only clutered ones)            
            n_TI_total= self.df_TI_props[self.df_TI_props[cluster] != -1].shape[0]
            
            # how many models to select
            if n_TIs > n_TI_total:
                n_TIs = n_TI_total

            #how many models in each cluster
            TI_cluster_dist = Counter(self.df_TI_props[self.df_TI_props[cluster] != -1][cluster])
            TI_per_cluster = []
            #depending on cluster size, determine how many models to pick per cluster
            for i in range(len(TI_cluster_dist)):
                TI_per_cluster.append(np.round(TI_cluster_dist[i]/n_TI_total*n_TIs))
            # check and correct if too many/too few models selected
            
            if np.sum(TI_per_cluster) == n_TIs:
                pass
            elif np.sum(TI_per_cluster) > n_TIs:
                while np.sum(TI_per_cluster) != n_TIs:
                    TI_per_cluster[TI_per_cluster.index(max(TI_per_cluster))] =  TI_per_cluster[TI_per_cluster.index(max(TI_per_cluster))] -1
            elif np.sum(TI_per_cluster) < n_TIs:
                while np.sum(TI_per_cluster) != n_TIs:
                    TI_per_cluster[TI_per_cluster.index(min(TI_per_cluster))] =  TI_per_cluster[TI_per_cluster.index(min(TI_per_cluster))] +1

            # randomly sample n models according to TI_per_cluster from each clusters
            TI_sampler = self.df_TI_props[self.df_TI_props[cluster] != -1].copy()
            TI_sampler["index_1"] = TI_sampler.index
            column_names = TI_sampler.columns.values.tolist()
            self.df_best_TIs_to_save = pd.DataFrame(columns = column_names )

            for i in range(len(TI_per_cluster)):
                for j in range(int(TI_per_cluster[i])):
                    sample_from_cluster = TI_sampler[(TI_sampler[cluster] == i)].sample()
                    #drop the model that has just been sampled
                    TI_sampler.drop(sample_from_cluster.index_1,inplace = True)
                    self.df_best_TIs_to_save = pd.concat([self.df_best_TIs_to_save,sample_from_cluster])
            
            if include_unclustered == True:
                # get models that arent clustered and append them
                if -1 is not self.df_TI_props[cluster]:
                    self.df_best_TIs_to_save = self.df_best_TIs_to_save.append(self.df_TI_props[self.df_TI_props[cluster] == -1])
        
        elif manual_pick == True:
        # feed the manual_pick list and select models acorndignly
            TI_sampler = self.df_TI_props.copy()
            TI_sampler["index_1"] = TI_sampler.index
            column_names = TI_sampler.columns.values.tolist()
            self.df_best_TIs_to_save = pd.DataFrame(columns = column_names )

            for i in range(manual_TI_id_list):
                sample = TI_sampler[(TI_sampler.index_1 == manual_TI_id_list[i])]
                self.df_best_TIs_to_save = pd.concat([self.df_best_TIs_to_save,sample])

        # show selected models with all models
        fig = make_subplots(rows = 1, cols = 1)

        cluster_x = "cluster_{}_x".format(cluster_parameter)
        cluster_y = "cluster_{}_y".format(cluster_parameter)


        fig.add_trace(go.Scatter(x = self.df_TI_props[cluster_x],
                                    y = self.df_TI_props[cluster_y],
                                mode='markers',
                                text = self.df_TI_props.index,
                                marker=dict(
                                            size=16,
                                            color=self.df_TI_props[cluster], #set color equal to a variable
                                            colorscale= "deep",#'Viridis', # one of plotly colorscales
                                            showscale=True,
                                            colorbar=dict(title="Clusters")
                                            )
                                ))
        fig.add_trace(go.Scatter(x = self.df_best_TIs_to_save[cluster_x],
                                    y = self.df_best_TIs_to_save[cluster_y],
                                    mode='markers',
                                    text = self.df_TI_props.index,
                                    marker = dict(
                                                size = 14,
                                                color = 0,
                                                )
                                ))
                        
        fig.update_layout(showlegend=False)
        fig.update_layout(title='Clustering of {} best models - Number of clusters found: {} - Unclustered models: {}'.format(self.df_TI_props.shape[0],self.df_TI_props[cluster].max()+1,abs(self.df_TI_props[cluster][self.df_TI_props[cluster] == -1].sum())))
        fig.show()

    def save_best_clustered_TIs(self):
        """ save the best,previously selected reservoir models that are ready for 
            also generate 2 csv files with df best_position of selected and all best models. """        

        path = str(self.base_path / "../../Output/training_images/{}/".format(self.TI_name) / self.dataset / "_")[:-1] # add dummz "_" compoenent athen stirp it to get final backslash
        print("Path {}".format(path))
        n_best_TI_selected = len(self.df_best_TIs_to_save)
        best_TI_selected_index = self.df_best_TIs_to_save.index.tolist()

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
            self.df_best_TIs_to_save.to_csv(best_position_path,index=False)
            # self.df_best_position.to_csv(best_position_path_2,index = False)

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
            for i in range(n_best_TI_selected):
                
                data_file_all_path = data_all_path + "TI{}.DATA".format(best_TI_selected_index[i])  
                permx_file_all_path = permx_all_path + "TI{}.GRDECL".format(best_TI_selected_index[i])  
                permy_file_all_path = permy_all_path + "TI{}.GRDECL".format(best_TI_selected_index[i])  
                permz_file_all_path = permz_all_path + "TI{}.GRDECL".format(best_TI_selected_index[i])  
                poro_file_all_path = poro_all_path + "TI{}.GRDECL".format(best_TI_selected_index[i])

                data_file_best_path = data_best_path + "TI{}.DATA".format(best_TI_selected_index[i])  
                permx_file_best_path = permx_best_path + "TI{}.GRDECL".format(best_TI_selected_index[i])  
                permy_file_best_path = permy_best_path + "TI{}.GRDECL".format(best_TI_selected_index[i])  
                permz_file_best_path = permz_best_path + "TI{}.GRDECL".format(best_TI_selected_index[i])  
                poro_file_best_path = poro_best_path + "TI{}.GRDECL".format(best_TI_selected_index[i]) 

                shutil.copy(data_file_all_path,data_file_best_path)
                shutil.copy(permx_file_all_path,permx_file_best_path)
                shutil.copy(permy_file_all_path,permy_file_best_path)
                shutil.copy(permz_file_all_path,permz_file_best_path)
                shutil.copy(poro_file_all_path,poro_file_best_path)

                # save best position df as csv. the selected ones and all best models
                best_position_path = destination_best_path + "best_models_selected.csv"
                # best_position_path_2 = destination_best_path + "best_models.csv"
                # best_tof_path = destination_best_path + "best_models_tof.csv"

                self.df_best_TIs_to_save.to_csv(best_position_path,index=False)
                # self.df_best_position.to_csv(best_position_path_2,index = False)
                # self.df_best_tof.to_csv(best_tof_path,index = False)


# what do I need? load TI tof into this class. do this by either loading in setup or just using same setup file. 
# then use tof clustering on them as done in postprocessing
# then decide how many training images I want from each cluster.
# save these training images in new folder, including the settings to create them. 
# also make entropy map of tof
# what can then be done is use porothresh as a cutoff for if flow through fracture or flow through matrix happens. this should be used by the pso.  


