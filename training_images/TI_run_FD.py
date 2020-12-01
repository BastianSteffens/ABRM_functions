import multiprocessing as mp
import matlab.engine
import numpy as np
import pandas as pd
import bz2
import _pickle as cPickle
import os

class TI_run_FD():
  """ class that runs Flow diagnostics on all models in current_run folder and outputs results into datetime folder with results """

  def __init__(self,setup):
    """ initialize class and load in dataset that will be run """ 

    # set base path where data is stored
    self.setup = setup

    self.n_TI = self.setup["n_TI"]
    self.pool = self.setup["pool"]
    self.n_shedules = self.setup["n_shedules"]

    # self.all_TI_performance = pd.DataFrame(columns = ["EV","tD","F","Phi","LC","tof","TI_no"])
    self.all_TI_performance = pd.DataFrame()

  def TI_run_FD_runner(self):

    if self.setup["n_shedules"] == 1:
      if self.pool is None:
        self.run_FD_iterator()

      else:
        pool = mp.Pool(self.pool)
        TI_array = np.arange(0,self.n_TI)
        # particle_list = pool.map(all_particles.calculate_particle_parallel,[particle_no for particle_no in particle_array])
        print("running parallel")

        TI_list = pool.map(self.run_FD_parallel,[TI_no for TI_no in TI_array])
        for TI_no in range(self.n_TI):
            TI_performance = TI_list[TI_no]["TI_performance"]
            self.all_TI_performance = self.all_TI_performance.append(TI_performance,ignore_index = True)
    
    else:
      if self.pool is None:
        print("running with multiple shedules")
        self.run_FD_iterator_multi()

      else:
        pool = mp.Pool(self.pool)
        TI_array = np.arange(0,self.n_TI)
        # particle_list = pool.map(all_particles.calculate_particle_parallel,[particle_no for particle_no in particle_array])
        print("running parallel with multiple shedules")

        TI_list = pool.map(self.run_FD_parallel_multi,[TI_no for TI_no in TI_array])
        for TI_no in range(self.n_TI):
            TI_performance = TI_list[TI_no]["TI_performance"]
            self.all_TI_performance = self.all_TI_performance.append(TI_performance,ignore_index = True)

    self.get_output_dfs()

    self.save_data()

  def run_FD(self,TI_no):

      eng = matlab.engine.start_matlab()
      # run matlab and mrst
      eng.matlab_starter(nargout = 0)

      # run FD and output dictionary
      TI_no = TI_no if self.pool is None else TI_no.item()
      FD_data = eng.FD_TI(TI_no)

      # split into Ev tD F Phi and LC and tof column
      FD_data = np.array(FD_data._data).reshape((6,len(FD_data)//6))
      TI_performance = pd.DataFrame()
      
      TI_performance["EV"] = FD_data[0]
      TI_performance["tD"] = FD_data[1]
      TI_performance["F"] = FD_data[2]
      TI_performance["Phi"] = FD_data[3]
      TI_performance["LC"] = FD_data[4]
      TI_performance["tof"] = FD_data[5]
      TI_performance = TI_performance.astype("float32")

      return(TI_performance)
  
  def run_FD_multi(self,TI_no,shedule_no):

        eng = matlab.engine.start_matlab()
        # run matlab and mrst
        eng.matlab_starter(nargout = 0)

        # run FD and output dictionary
        # TI_no = TI_no if self.pool is None else TI_no.item()

        TI_name = str(TI_no) + "_" + str(shedule_no)
        FD_data = eng.FD_TI(TI_name)

        # split into Ev tD F Phi and LC and tof column
        FD_data = np.array(FD_data._data).reshape((6,len(FD_data)//6))
        TI_performance = pd.DataFrame()

        EV = "EV_" + str(shedule_no)
        tD = "tD_" + str(shedule_no)
        F = "F_" + str(shedule_no)
        Phi = "Phi_" + str(shedule_no)
        LC = "LC_" + str(shedule_no)
        tof = "tof_" + str(shedule_no)
        # tof_for = "tof_for_" + str(shedule_no)
        # tof_back = "tof_back_" + str(shedule_no)
        # tof_combi = "tof_combi_" + str(shedule_no)
        # prod_part = "prod_part_" + str(shedule_no)
        # inj_part = "inj_part_" + str(shedule_no)
        
        TI_performance[EV] = FD_data[0]
        TI_performance[tD] = FD_data[1]
        TI_performance[F] = FD_data[2]
        TI_performance[Phi] = FD_data[3]
        TI_performance[LC] = FD_data[4]
        TI_performance[tof] = FD_data[5]
        # TI_performance[tof_back] = FD_data[6]
        # TI_performance[tof_combi] = FD_data[7]
        # TI_performance[prod_part] = FD_data[8]
        # TI_performance[inj_part] = FD_data[9]

        TI_performance = TI_performance.astype("float32")

        return(TI_performance)

  def run_FD_parallel(self,TI_no):
      """run flow diagnostics parallel """ 
      TI_performance = self.run_FD(TI_no)

      TI_no = TI_no if self.pool is None else TI_no.item()

      TI_performance["TI_no"] = TI_no
      TI_dict = dict()
      TI_dict["TI_performance"] = TI_performance
      print("{}/1200 TIs done".format(TI_no))
      return TI_dict

  def run_FD_parallel_multi(self,TI_no):
    """run flow diagnostics parallel """
    TI_performance = pd.DataFrame()
    for shedule_no in range(self.n_shedules):

        self.built_FD_Data_files_multi(TI_no= TI_no,shedule_no= shedule_no)
        TI_performance_current_shedule = self.run_FD_multi(TI_no,shedule_no)

        # TI_no = TI_no if self.pool is None else TI_no.item()

        TI_performance["TI_no"] = TI_no
        EV = "EV_" + str(shedule_no)
        tD = "tD_" + str(shedule_no)
        F = "F_" + str(shedule_no)
        Phi = "Phi_" + str(shedule_no)
        LC = "LC_" + str(shedule_no)
        tof = "tof_" + str(shedule_no)

        TI_performance[EV] = TI_performance_current_shedule[EV]
        TI_performance[tD] = TI_performance_current_shedule[tD] 
        TI_performance[F] = TI_performance_current_shedule[F]
        TI_performance[Phi] = TI_performance_current_shedule[Phi]
        TI_performance[LC] = TI_performance_current_shedule[LC]
        TI_performance[tof] = TI_performance_current_shedule[tof]


    TI_dict = dict()
    TI_dict["TI_performance"] = TI_performance
    print("{}/1200 TIs done".format(TI_no))
    return TI_dict

  def run_FD_iterator(self):
      """run flow diagnostics one by one """ 

      for TI_no in range(self.n_TI):

        TI_performance = self.run_FD(TI_no)

        self.all_TI_performance = self.all_TI_performance.append(TI_performance) # store for data saving
  
  def run_FD_iterator_multi(self):
      """run flow diagnostics one by one on multiple shedules """ 

      for TI_no in range(self.n_TI):
          for shedule_no in range(self.n_shedules):
            self.built_FD_Data_files_multi(TI_no= TI_no,shedule_no= shedule_no)
            TI_performance = self.run_FD_multi(TI_no,shedule_no)

            self.all_TI_performance = self.all_TI_performance.append(TI_performance) # store for data saving
  
  def get_output_dfs(self):
    """prepare dfs for output that is ready for postprocessing"""
    
    if self.setup["n_shedules"] == 1:

      # raw data from FD
      self.tof = self.all_TI_performance[["tof","TI_no"]].copy()
      # shorten performance dataset to save time and space. this should be enough for plotting
      self.all_TI_performance_short = self.all_TI_performance.iloc[::100,:].copy()
      
    else:

      self.tof = self.all_TI_performance[["TI_no"]].copy()
      for shedule_no in range(self.n_shedules):
        tof = "tof_" + str(shedule_no)
        self.tof[tof] = self.all_TI_performance[tof]

      self.all_TI_performance_short = self.all_TI_performance.iloc[::100,:].copy()

  def save_data(self):
      """ save df to csv files / pickle that contains all data used for postprocessing """

      # filepath setup
      folder_path = self.setup["folder_path"]
      
      output_file_performance = "all_TI_performance.pbz2"
      tof_file = "tof_all_TI.pbz2"

      file_path_tof = folder_path / tof_file
      file_path_performance = folder_path / output_file_performance

      # make folder
      if not os.path.exists(folder_path):
          os.makedirs(folder_path)
      
      # save all
      with bz2.BZ2File(file_path_tof,"w") as f:
          cPickle.dump(self.tof,f, protocol= 4)
      with bz2.BZ2File(file_path_performance,"w") as f:
          cPickle.dump(self.all_TI_performance_short,f, protocol= 4)

  def built_FD_Data_files_multi(self,TI_no,shedule_no):
      # loading in settings that I set up on init_ABRM.py for this run

      # schedule = self.setup["schedule_multi_"+str(shedule_no)]
      shedule = "SCHEDULE_multi_" + str(shedule_no)

      data_file = "RUNSPEC\n\nTITLE\nModel_{}_{}\n\nDIMENS\n--NX NY NZ\n60 60 1 /\n\n--Phases\nOIL\nWATER\n\n--DUALPORO\n--NODPPM\n\n--Units\nMETRIC\n\n--Number of Saturation Tables\nTABDIMS\n1 /\n\n--Maximum number of Wells\nWELLDIMS\n10 100 5 10 /\n\n--First Oil\nSTART\n1 OCT 2017 /\n\n--Memory Allocation\nNSTACK\n100 /\n\n--How many warnings allowed, but terminate after first error\nMESSAGES\n11*5000 1 /\n\n--Unified Output Files\nUNIFOUT\n\n--======================================================================\n\nGRID\n--Include corner point geometry model\nINCLUDE\n'..\INCLUDE\GRID.GRDECL'\n/\n\nACTNUM\n3600*1 /\n\n--Porosity\nINCLUDE\n'..\INCLUDE\PORO\TI{}.GRDECL'\n/\n\n--Permeability\nINCLUDE\n'..\INCLUDE\PERMX\TI{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMY\TI{}.GRDECL'\n/\nINCLUDE\n'..\INCLUDE\PERMZ\TI{}.GRDECL'\n/\n\n--Net to Gross\nNTG\n3600*1\n/\n\n--Output .INIT file to allow viewing of grid data in post proessor\nINIT\n\n--======================================================================\n\nPROPS\n\nINCLUDE\n'..\INCLUDE\DP_pvt.inc' /\n\nINCLUDE\n'..\INCLUDE\ROCK_RELPERMS.INC' /\n\n--======================================================================\n\nREGIONS\n\nEQLNUM\n3600*1\n/\nSATNUM\n3600*1\n/\nPVTNUM\n3600*1\n/\n\n--======================================================================\n\nSOLUTION\n\nINCLUDE\n'..\INCLUDE\SOLUTION.INC' /\n\n--======================================================================\n\nSUMMARY\n\nINCLUDE\n'..\INCLUDE\SUMMARY.INC' /\n\n--======================================================================\n\nSCHEDULE\n\nINCLUDE\n'..\INCLUDE\{}.INC' /\n\nEND".format(TI_no,shedule_no,TI_no,TI_no,TI_no,TI_no,shedule)  

      # data_file_path = self.setup["base_path"] / "../FD_Models/DATA/TI{}_{}.DATA".format(TI_no,shedule_no)
      data_file_path = self.setup["base_path"] / "../../Output/training_images/current_run/DATA/TI{}_{}.DATA".format(TI_no,shedule_no)
      file = open(data_file_path, "w+")
      # write petrelfilepath and licence part into file and seed
      file.write(data_file)

      # close file
      file.close()