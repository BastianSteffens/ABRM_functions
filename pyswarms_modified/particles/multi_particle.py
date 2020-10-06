
########################
import numpy as np
import pandas as pd
import matlab.engine
from scipy import interpolate
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from GRDECL_file_reader.GRDECL2VTK import *
from geovoronoi import voronoi_regions_from_coords

########################

class multi_particle():
    """ class that calculates multi objective function particle """
    
    def __init__(self,swarm,setup,iteration):

        # set base path where data is stored
        self.setup = setup
        self.pool = self.setup["pool"]
        self.n_shedules = self.setup["n_shedules"]
        self.n_particles = self.setup["n_particles"]
        self.swarm = swarm
        self.iteration = iteration
        self.misfit_swarm = np.zeros((self.n_particles,self.n_shedules))
        self.LC_swarm = np.zeros((self.n_particles,self.n_shedules))
        self.swarm_performance = pd.DataFrame()

        print("########################################## starting model evaluation  iteration {}/{} ##########################################".format(self.iteration,self.setup["n_iters"]-1))

    def calculate_particle_performance_multi(self,particle_no,shedule_no):
        # Objective Function run flow diagnostics
        particle_performance = self.obj_fkt_FD_multi(particle_no,shedule_no)

        # Compute Performance
        particle_misfit = self.misfit_fkt_F_Phi_curve_multi(particle_performance,shedule_no)
        print('particle {}/{} - misfit {}'.format(particle_no,self.setup["n_particles"]-1,np.round(particle_misfit,3)),end = "\r")

        # store misfit and particle no and iteration in dataframe
        particle_no = particle_no if self.pool is None else particle_no.item()

        particle_performance["iteration"] =self.iteration
        particle_performance["particle_no"] = particle_no
        misfit = "misfit_" + str(shedule_no)
        particle_performance[misfit] = particle_misfit

        return particle_performance

    def obj_fkt_FD_multi(self,particle_no,shedule_no):

        eng = matlab.engine.start_matlab()
        # run matlab and mrst
        eng.matlab_starter(nargout = 0)

        # run FD and output dictionary
        particle_no = particle_no if self.pool is None else particle_no.item()

        model_name = str(particle_no) + "_" + str(shedule_no)
        FD_data = eng.FD_BS(model_name)

        # split into Ev tD F Phi and LC and tof column
        FD_data = np.array(FD_data._data).reshape((6,len(FD_data)//6))
        particle_performance = pd.DataFrame()

        EV = "EV_" + str(shedule_no)
        tD = "tD_" + str(shedule_no)
        F = "F_" + str(shedule_no)
        Phi = "Phi_" + str(shedule_no)
        LC = "LC_" + str(shedule_no)
        tof = "tof_" + str(shedule_no)
        
        particle_performance[EV] = FD_data[0]
        particle_performance[tD] = FD_data[1]
        particle_performance[F] = FD_data[2]
        particle_performance[Phi] = FD_data[3]
        particle_performance[LC] = FD_data[4]
        particle_performance[tof] = FD_data[5]
        particle_performance = particle_performance.astype("float32")

        return(particle_performance)

    def misfit_fkt_F_Phi_curve_multi(self,particle_performance,shedule_no):

    
        F = "F_" + str(shedule_no)
        Phi = "Phi_" + str(shedule_no)
 

        F_points_target = self.setup["F_points_target"]
        Phi_points_target = self.setup["Phi_points_target"]
        # interpolate F-Phi curve from imput points with spline
        tck = interpolate.splrep(Phi_points_target,F_points_target, s = 0)
        Phi_interpolated = np.linspace(0,1,num = len(particle_performance[Phi]),endpoint = True)
        F_interpolated = interpolate.splev(Phi_interpolated,tck,der = 0) # here can easily get first and second order derr.

        # calculate first order derivate of interpolated F-Phi curve and modelled F-Phi curve
        F_interpolated_first_derr = np.gradient(F_interpolated)
        F_first_derr = np.gradient(particle_performance[F])
        F_interpolated_second_derr = np.gradient(F_interpolated_first_derr)
        F_second_derr = np.gradient(F_first_derr)

        # calculate LC for interpolatd F-Phi curve and modelled F-Phi curve
        LC_interpolated = self.compute_LC(F_interpolated,Phi_interpolated)

        LC = self.compute_LC(particle_performance[F],particle_performance[Phi])
        # calculate rmse for each curve and LC
        rmse_0 = mean_squared_error(F_interpolated,particle_performance[F],squared=False)
        rmse_1 = mean_squared_error(F_interpolated_first_derr,F_first_derr,squared=False)
        rmse_2 = mean_squared_error(F_interpolated_second_derr,F_second_derr,squared=False)
        LC_error = abs(LC-LC_interpolated)

        # calculate misfit - RMSE if i turn squared to True it will calculate MSE
        misfit = rmse_0 + rmse_1 + rmse_2 + LC_error
        misfit = misfit.astype("float32")

        return misfit

    def compute_LC(self,F,Phi):

        v = np.diff(Phi,1)
        
        LC = 2*(np.sum(((np.array(F[0:-1]) + np.array(F[1:]))/2*v))-0.5)
        LC = LC.astype("float32")
        return LC
     
    def particle_iterator_multi(self):
        
        for particle_no in range(self.n_particles):
        
            # if working with voronoi tesselation for zonation. now its time to patch the previously built models together
            if self.setup["n_voronoi"] > 0:
                self.patch_voronoi_models(particle_no)

            for shedule_no in range(self.n_shedules):
                #built model for FD
                self.built_FD_Data_files_multi(particle_no,shedule_no)

                particle_performance = self.calculate_particle_performance_multi(particle_no,shedule_no)
                misfit = "misfit_" + str(shedule_no)
                LC = "LC_" + str(shedule_no)
                self.misfit_swarm[particle_no,shedule_no] = particle_performance[misfit][0]
                self.LC_swarm[particle_no,shedule_no] = particle_performance[LC][0]
                self.swarm_performance = self.swarm_performance.append(particle_performance) # store for data saving
                
        print('swarm misfit {}                  '.format(np.round(self.misfit_swarm,2)))

    def calculate_particle_parallel_multi(self,particle_no):

        # if working with voronoi tesselation for zonation. now its time to patch the previously built models together
        if self.setup["n_voronoi"] > 0:
            self.patch_voronoi_models(particle_no)

        for shedule_no in range(self.n_shedules):
            #built model for FD
            self.built_FD_Data_files_multi(particle_no,shedule_no)

            # Objective Function run flow diagnostics
            particle_performance = self.obj_fkt_FD_multi(particle_no,shedule_no)

            # Compute Performance
            particle_misfit = self.misfit_fkt_F_Phi_curve_multi(particle_performance,shedule_no)
            print('particle {}/{} - misfit {}'.format(particle_no,self.setup["n_particles"]-1,np.round(particle_misfit,3)))#,end = "\r")

            # store misfit and particle no and iteration in dataframe also voronoi zone assignment, if voronoi is used.
            particle_no = particle_no if self.pool is None else particle_no.item()

            particle_performance["iteration"] =self.iteration
            particle_performance["particle_no"] = particle_no
            misfit = "misfit_" + str(shedule_no)
            particle_performance[misfit] = particle_misfit

        # have to do this for parallel execution. can only return one argument. and need to return voroni stuff and particle_performacne
        particle_dict = dict()
        particle_dict["particle_performance"] = particle_performance
        if self.setup["n_voronoi"] > 0:
            particle_dict["assign_voronoi_zone_" + str(particle_no)] = self.setup["assign_voronoi_zone_" +str(particle_no)] 

        return particle_dict
    
    def built_FD_Data_files_multi(self,particle_no,shedule_no):
        # loading in settings that I set up on init_ABRM.py for this run

        # schedule = self.setup["schedule_multi_"+str(shedule_no)]
        shedule = "SCHEDULE_multi_" + str(shedule_no)

        data_file = "RUNSPEC\n\nTITLE\nModel_{}_{}\n\nDIMENS\n--NX NY NZ\n200 100 7 /\n\n--Phases\nOIL\nWATER\n\n--DUALPORO\n--NODPPM\n\n--Units\nMETRIC\n\n--Number of Saturation Tables\nTABDIMS\n1 /\n\n--Maximum number of Wells\nWELLDIMS\n10 100 5 10 /\n\n--First Oil\nSTART\n1 OCT 2017 /\n\n--Memory Allocation\nNSTACK\n100 /\n\n--How many warnings allowed, but terminate after first error\nMESSAGES\n11*5000 1 /\n\n--Unified Output Files\nUNIFOUT\n\n--======================================================================\n\nGRID\n--Include corner point geometry model\nINCLUDE\n'..\INCLUDE\GRID.GRDECL'\n/\n\nACTNUM\n140000*1 /\n\n--Porosity\nINCLUDE\n'..\INCLUDE\PORO\M{}.GRDECL'\n/\n\n--Permeability\nINCLUDE\n'..\INCLUDE\PERMX\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMY\M{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMZ\M{}.GRDECL'\n/\n\n--Net to Gross\nNTG\n140000*1\n/\n\n--Output .INIT file to allow viewing of grid data in post proessor\nINIT\n\n--======================================================================\n\nPROPS\n\nINCLUDE\n'..\INCLUDE\DP_pvt.inc' /\n\nINCLUDE\n'..\INCLUDE\ROCK_RELPERMS.INC' /\n\n--======================================================================\n\nREGIONS\n\nEQLNUM\n140000*1\n/\nSATNUM\n140000*1\n/\nPVTNUM\n140000*1\n/\n\n--======================================================================\n\nSOLUTION\n\nINCLUDE\n'..\INCLUDE\SOLUTION.INC' /\n\n--======================================================================\n\nSUMMARY\n\nINCLUDE\n'..\INCLUDE\SUMMARY.INC' /\n\n--======================================================================\n\nSCHEDULE\n\nINCLUDE\n'..\INCLUDE\{}.INC' /\n\nEND".format(particle_no,shedule_no,particle_no,particle_no,particle_no,particle_no,shedule)  
        data_file_path = self.setup["base_path"] / "../FD_Models/DATA/M_FD_{}_{}.DATA".format(particle_no,shedule_no)

        file = open(data_file_path, "w+")
        # write petrelfilepath and licence part into file and seed
        file.write(data_file)

        # close file
        file.close()

    def patch_voronoi_models(self,particle_no):

        n_voronoi = self.setup["n_voronoi"]
        n_voronoi_zones = self.setup["n_voronoi_zones"]
        parameter_type = self.setup["parameter_type"]
        n_parameters = self.setup["n_parameters"]
        parameter_name = self.setup["columns"]
        nx = self.setup["nx"]
        ny = self.setup["ny"]
        nz = self.setup["nz"]
        iter_ticker = self.iteration

        n_neighbors = np.int(n_voronoi /n_voronoi_zones)
        
        # first figure out which points I am interested and append them to new list
        voronoi_x = []
        voronoi_y = []
        # voronoi_z = []
        for j in range(n_parameters):

            # find voronoi positions
            if parameter_type[j] == 3:
                if "x" in parameter_name[j]:
                    voronoi_x_temp = self.swarm.position_converted[particle_no,j]
                    voronoi_x.append(voronoi_x_temp)
                elif "y" in parameter_name[j]:
                    voronoi_y_temp = self.swarm.position_converted[particle_no,j]
                    voronoi_y.append(voronoi_y_temp)
                # elif "z" in parameter_name[j]:
                #     voronoi_z_temp = self.swarm.position_converted[particle_no,j]
                #     voronoi_z.append(voronoi_z_temp)

        # use these points to built a voronoi tesselation
        voronoi_x = np.array(voronoi_x)
        voronoi_y = np.array(voronoi_y)
        voronoi_points = np.vstack((voronoi_x,voronoi_y)).T
        # voronoi_z = np.array(voronoi_z)

        #crosscheck if any points lie on top of each other. if so --> move one
        for j in range(len(voronoi_points)):
            boolean = voronoi_points == voronoi_points[j]
            dublicate_tracker = 0
            for k in range(len(boolean)):
                if np.sum(boolean[k]) == 2:
                    dublicate_tracker +=1
                if dublicate_tracker ==2:
                    print("dublicate voronoi points --> reassignemnt")
                    voronoi_points[k] = voronoi_points[k]+ np.random.randint(0,10)

                    if voronoi_points[k,0] >= nx:
                        voronoi_points[k,0] = voronoi_points[k,0] - np.random.randint(0,20)
                    if voronoi_points[k,1] >= ny:
                        voronoi_points[k,1] = voronoi_points[k,1] - np.random.randint(0,20)

                    dublicate_tracker = 0

        # #define grid and  position initianinon points of n polygons
        grid = Polygon([(0, 0), (0, ny), (nx, ny), (nx, 0)])

        # generate 2D mesh
        x = np.arange(0,nx+1,1,)
        y = np.arange(0,ny+1,1)
        x_grid, y_grid = np.meshgrid(x,y)

        #get cell centers of mesh
        x_cell_center = x_grid[:-1,:-1]+0.5
        y_cell_center = y_grid[:-1,:-1]+0.5

        cell_center_which_polygon = np.zeros(len(x_cell_center.flatten()))

        # array to assign polygon id [ last column] to cell id [first 2 columns]
        all_cell_center = np.column_stack((x_cell_center.flatten(),y_cell_center.flatten(),cell_center_which_polygon))

        # get voronoi regions
        poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(voronoi_points, grid,farpoints_max_extend_factor = 30)

        # assign cells to a zone in first iteration stick to taht assingment.
        if iter_ticker == 0:

            # find centroids of vornoi polygons
            voronoi_centroids = []
            for j in range(n_voronoi):
                voronoi_centroids.append(np.array(poly_shapes[j].centroid))
            voronoi_centroids = np.array(voronoi_centroids)

            # assign each vornoi polygone to one of the n voronoi polygon zones with KNN
            knn = NearestNeighbors(n_neighbors= n_neighbors, algorithm='auto',p=2)

            assign_voronoi_zone = np.empty(n_voronoi)
            assign_voronoi_zone[:] = np.nan
            points_to_pick_from = voronoi_centroids
            for j in range(n_voronoi_zones):

                #randomly pick starting point of zone
                init_point = np.random.choice(len(points_to_pick_from))
                # find nearest points for starting point zone
                knn.fit(points_to_pick_from)
                _, indices = knn.kneighbors(points_to_pick_from[init_point].reshape(1,-1))

                for k in range(n_neighbors):    
                    # assing these points to a zone
                    assigner = np.where(voronoi_centroids == points_to_pick_from[indices[0,k]],1,0)
                    assigner = np.sum(assigner,axis = 1)
                    assigner = np.where(assigner ==2)
                    assign_voronoi_zone[assigner[0][0]] = j
                # remove selected points from array to choose from
                points_to_pick_from = np.delete(points_to_pick_from,indices,axis = 0)


            self.setup["assign_voronoi_zone_" +str(particle_no)] = assign_voronoi_zone

        else:
        # load voronoi zone assignemnt
            assign_voronoi_zone = self.setup["assign_voronoi_zone_" + str(particle_no)]
        # in what voronoi zone and vornoi polygon do cell centers plot

        for j in range(len(all_cell_center)):
            for voronoi_polygon_id in range(n_voronoi):
                
                polygon = poly_shapes[voronoi_polygon_id]
                cell_id = Point(all_cell_center[j,0],all_cell_center[j,1])
                
                if polygon.intersects(cell_id):
                    all_cell_center[j,2] = assign_voronoi_zone[voronoi_polygon_id]
        
        # load and assign correct grdecl files to each polygon zone and patch togetehr to new model
        #output from reservoir modelling
        cell_vornoi_combination = np.tile(all_cell_center[:,2],nz).reshape((nx,ny,nz))
        cell_vornoi_combination_flatten = cell_vornoi_combination.flatten()

        all_model_values_permx = np.zeros((n_voronoi_zones,len(cell_vornoi_combination_flatten)))
        all_model_values_permy = np.zeros((n_voronoi_zones,len(cell_vornoi_combination_flatten)))
        all_model_values_permz = np.zeros((n_voronoi_zones,len(cell_vornoi_combination_flatten)))
        all_model_values_poro = np.zeros((n_voronoi_zones,len(cell_vornoi_combination_flatten)))

        geomodel_path = str(self.setup["base_path"] / "../FD_Models/INCLUDE/GRID.grdecl")
        Model = GeologyModel(filename = geomodel_path)
        data_file_path = self.setup["base_path"] / "../FD_Models/DATA/M_FD_{}.DATA".format(particle_no)

        for j in range(n_voronoi_zones):
            temp_model_path_permx = self.setup["base_path"] / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PERMX/M{}.GRDECL'.format(j,particle_no)
            temp_model_path_permy = self.setup["base_path"] / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PERMY/M{}.GRDECL'.format(j,particle_no)
            temp_model_path_permz = self.setup["base_path"] / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PERMZ/M{}.GRDECL'.format(j,particle_no)
            temp_model_path_poro = self.setup["base_path"] / '../FD_Models/INCLUDE/Voronoi/Patch_{}/PORO/M{}.GRDECL'.format(j,particle_no)
            temp_model_permx = Model.LoadCellData(varname="PERMX",filename=temp_model_path_permx)
            temp_model_permy = Model.LoadCellData(varname="PERMY",filename=temp_model_path_permy)
            temp_model_permz = Model.LoadCellData(varname="PERMZ",filename=temp_model_path_permz)
            temp_model_poro = Model.LoadCellData(varname="PORO",filename=temp_model_path_poro)

            all_model_values_permx[j] = temp_model_permx
            all_model_values_permy[j] = temp_model_permy
            all_model_values_permz[j] = temp_model_permz
            all_model_values_poro[j] = temp_model_poro

        # patch things together
        patch_permx  = []
        patch_permy  = []
        patch_permz  = []
        patch_poro  = []
        for j in range(len(cell_vornoi_combination_flatten)):
            for k in range(n_voronoi_zones):
            
                if cell_vornoi_combination_flatten[j] == k:
                    permx = all_model_values_permx[k,j]
                    permy = all_model_values_permy[k,j]
                    permz = all_model_values_permz[k,j]
                    poro = all_model_values_poro[k,j]
                    patch_permx.append(permx)
                    patch_permy.append(permy)
                    patch_permz.append(permz)
                    patch_poro.append(poro)    


        file_permx_beginning = "FILEUNIT\nMETRIC /\n\nPERMX\n"
        permx_file_path = self.setup["base_path"] / "../FD_Models/INCLUDE/PERMX/M{}.GRDECL".format(particle_no)
        patch_permx[-1] = "{} /".format(patch_permx[-1])
        with open(permx_file_path,"w+") as f:
            f.write(file_permx_beginning)
            newline_ticker = 0
            for item in patch_permx:
                newline_ticker += 1
                if newline_ticker == 50:
                    f.write("\n")
                    newline_ticker = 0
                f.write("{} ".format(item))
            f.close()

        file_permy_beginning = "FILEUNIT\nMETRIC /\n\nPERMY\n"
        permy_file_path = self.setup["base_path"] / "../FD_Models/INCLUDE/PERMY/M{}.GRDECL".format(particle_no)
        patch_permy[-1] = "{} /".format(patch_permy[-1])
        with open(permy_file_path,"w+") as f:
            f.write(file_permy_beginning)
            newline_ticker = 0
            for item in patch_permy:
                newline_ticker += 1
                if newline_ticker == 50:
                    f.write("\n")
                    newline_ticker = 0
                f.write("{} ".format(item))
            f.close()

        file_permz_beginning = "FILEUNIT\nMETRIC /\n\nPERMZ\n"
        permz_file_path = self.setup["base_path"] / "../FD_Models/INCLUDE/PERMZ/M{}.GRDECL".format(particle_no)
        patch_permz[-1] = "{} /".format(patch_permz[-1])
        with open(permz_file_path,"w+") as f:
            f.write(file_permz_beginning)
            newline_ticker = 0
            for item in patch_permz:
                newline_ticker += 1
                if newline_ticker == 50:
                    f.write("\n")
                    newline_ticker = 0
                f.write("{} ".format(item))
            f.close()

        file_poro_beginning = "FILEUNIT\nMETRIC /\n\nPORO\n"
        poro_file_path = self.setup["base_path"] / "../FD_Models/INCLUDE/PORO/M{}.GRDECL".format(particle_no)
        patch_poro[-1] = "{} /".format(patch_poro[-1])
        with open(poro_file_path,"w+") as f:
            f.write(file_poro_beginning)
            newline_ticker = 0
            for item in patch_poro:
                newline_ticker += 1
                if newline_ticker == 50:
                    f.write("\n")
                    newline_ticker = 0
                f.write("{} ".format(item))
            f.close()

  