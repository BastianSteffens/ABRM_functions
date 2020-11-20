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

    self.all_TI_performance = pd.DataFrame(columns = ["EV","tD","F","Phi","LC","tof","TI_no"])

  def TI_run_FD_runner(self):

    if self.pool is None:
      self.run_FD_iterator()

    else:
      pool = mp.Pool(self.pool)
      TI_array = np.arange(0,self.n_TI)
      # particle_list = pool.map(all_particles.calculate_particle_parallel,[particle_no for particle_no in particle_array])

      TI_list = pool.map(self.run_FD_parallel,[TI_no for TI_no in TI_array])
      print("running parallel")
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


  def run_FD_parallel(self,TI_no):
      """run flow diagnostics parallel """ 
      TI_performance = self.run_FD(TI_no)

      TI_no = TI_no if self.pool is None else TI_no.item()

      TI_performance["TI_no"] = TI_no
      TI_dict = dict()
      TI_dict["TI_performance"] = TI_performance

      return TI_dict

  def run_FD_iterator(self):
      """run flow diagnostics one by one """ 

      for TI_no in range(self.n_TI):

          TI_performance = self.run_FD(TI_no)

          self.all_TI_performance = self.all_TI_performance.append(TI_performance) # store for data saving
    
  
  def get_output_dfs(self):
    """prepare dfs for output that is ready for postprocessing"""
    
    # raw data from FD
    self.tof = self.all_TI_performance[["tof","TI_no"]].copy()
    # shorten performance dataset to save time and space. this should be enough for plotting
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
