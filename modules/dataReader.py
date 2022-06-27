
class DataReader():
    
    def __init__(self, time):
        self.time = time
    
    
    def _getMockData(self):
        
        
    
    def getLatestRecords(self, time=None):
        
        if time is not None:
            self.time = time
            
        '''
            Get the latest records from IOTdb and convert the results into standard format
            and return the results 
            
        '''    