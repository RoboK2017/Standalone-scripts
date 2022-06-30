import json
import numpy as np
import os, shutil

class ConfigReader(dict):

    '''
    This class is used to read the config file for the ROI detector
    '''
    
    def __init__(self, config):
        with open(config) as c:
            
            conf = json.load(c)
            temp, vals = {}, []
            outlier_priority = conf['outlier_priority']
            rev_map = {0:'roi', outlier_priority :'person_in_forklift'}
            
            out_path = conf['output_folder']
            
            #self._cleanFolder(out_path)
            
            for x in conf['target_pair_nm']:
                k1 = x[0]+','+x[1]
                k2 = x[1]+','+x[0]
                vals.append(x[1])
                vals.append(x[0])
                rev_map[x[2]] = k1
                temp[k1], temp[k2] = x[2], x[2]
                
            
            conf['target_pair_nm'] = temp 
            conf['target_class_nm'] = np.unique(vals)
            conf['nm_rev_map'] = rev_map
            #rev_map[1] = ""
            
            for k in rev_map:
                out_fol = os.path.join(out_path, rev_map[k])
                if not os.path.exists(out_fol):
                    os.makedirs(out_fol)

           # o_f = os.path.join(out_path, "")       

            
            conf['roi'] = [np.array(reg, dtype=int) for reg in conf['roi']]
            conf['roi_nm'] = [np.array(reg, dtype=int) for reg in conf['roi_nm']]
            
        self.update(conf)
        
    def _cleanFolder(self, folder):
        
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))