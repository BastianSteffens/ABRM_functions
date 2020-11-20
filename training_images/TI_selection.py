class TI_selection():
  """ pick Training images wiht help of clustering that should be taken forward """
        
        def __init__(self,setup):
          
            self.setup = setup
            self.df_tof = pd.DataFrame()


# what do I need? load TI tof into this class. do this by either loading in setup or just using same setup file. 
# then use tof clustering on them as done in postprocessing
# then decide how many training images I want from each cluster.
# save these training images in new folder, including the settings to create them. 
# also make entropy map of tof
# what can then be done is use porothresh as a cutoff for if flow through fracture or flow through matrix happens. this should be used by the pso.  


