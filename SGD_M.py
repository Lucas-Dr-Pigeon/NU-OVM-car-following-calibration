# -*- coding: utf-8 -*-
import src.preprocessing as prep
import pandas as pd
import numpy as np 
import datetime
import random
import matplotlib.pyplot as plt
import math

''' a function allowing you to save data '''
def save_data(data, filepath):
    import pickle
    with open(filepath, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
   
''' a function allowing you to save data '''                 
def load_data(filepath):
    import pickle
    with open(filepath, 'rb') as fp:
        data = pickle.load(fp)
    return data

''' vehicle trajectory object '''
class Car():
    def __init__(self, loc, frame, vel, acc, lane):
        self.loc = loc
        self.frame = frame
        self.vel = vel
        self.acc = acc
        self.lane = lane

''' you can import and manipulate the dataset as an object '''
class Dataset():
    def __init__(self):
        ''' specify dataset file paths '''
        self.paths = [
            r"vehicle-trajectory-data\0750am-0805am\trajectories-0750am-0805am.csv",
            r"vehicle-trajectory-data\0805am-0820am\trajectories-0805am-0820am.csv",
            r"vehicle-trajectory-data\0820am-0835am\trajectories-0820am-0835am.csv"
            ]
        
    def __Import_Data__(self):
        data = []
        for path in self.paths:
            data.append(pd.read_csv(path))
        ''' store the trajectory records into the class'''
        self.data = pd.concat(data).reset_index(drop=True)
   
        ''' unix_time '''
        self.unix_time = np.asarray(sorted(set(self.data["Global_Time"])))
        
        ''' convert unix_time to date '''
        datetimes = []
        for t in self.unix_time:
            datetimes.append(self.__datetime__(t))
        self.datetimes = np.asarray(datetimes)  
        
        ''' report the study period '''
        self.period = ( self.unix_time[-1] - self.unix_time[0] ) / 100
        print ("Study Period: " + str(self.datetimes[0]) + " - " + str(self.datetimes[-1]))
        print ("Period Length: " + str( int( self.period /600 )) + " min " 
                                 + str( int( self.period %600 /10 )) + " s " 
                                 + str( int( self.period %600 %10 )) + " ms " )
        
        ''' report the study segment '''
        all_Y = np.asarray(sorted(set(self.data["Local_Y"])))
        self.segmentlength = all_Y[-1] - all_Y[0]
        print ("Study Segment Length: " + str(self.segmentlength) + " ft ")
        
        ''' store all the vehicles '''
        self.vehicles = np.asarray(sorted(set(self.data["Vehicle_ID"]))) 
        print ("Number of Vehicles: " + str(self.vehicles.shape[0]))
        
        ''' see how many lanes '''
        self.lanes = len(sorted(set(self.data["Lane_ID"]))) 
        print ("Number of Lanes: " + str(self.lanes))
        
        ''' return the data so that it could be called by another function '''
        return self.data

    def __datetime__(self, time):
        s = datetime.datetime.fromtimestamp(time/1000.0).strftime('%Y-%m-%d %H:%M:%S.%f')
        return s
    
    def __Dict_by_Frame__(self):
        _dict = {}
        _dict_v = {}
        for i in range(len(self.data)):
            print ("Processing record #" + str(i))
            trajectory = self.data[i:i+1]
            frame = trajectory["Global_Time"][i]
            loc = trajectory["Local_Y"][i]
            acc = trajectory["v_Acc"][i]
            vel = trajectory["v_Vel"][i]
            veh = trajectory["Vehicle_ID"][i]
            lane = trajectory["Lane_ID"][i]
            packet = { "loc": loc, "vel": vel, "acc": acc, "veh": veh, "lane": lane, "datetime": self.__datetime__(frame) }
            if lane not in _dict.keys():
                _dict[lane] = {}
            if frame not in _dict[lane].keys():
                _dict[lane][frame] = {}
            if veh not in _dict[lane][frame].keys():
                _dict[lane][frame][veh] = packet
            if veh not in _dict_v.keys():
                _dict_v[veh] = {}
            if frame not in _dict_v[veh].keys():
                _dict_v[veh][frame] = packet
        save_data(_dict, "./data/dictionary_frame")
        save_data(_dict_v, "./data/dictionary_vehicle")
        self.dict_by_frame = _dict
        self.dict_by_veh = _dict_v
        return _dict, _dict_v
    
    def __Sampling__(self, sample_size = 1e3, mpr = 100, period = 300, convoy_len = 600):
        cv = random.sample(set(self.vehicles), int(self.vehicles.shape[0] * mpr / 100 ))
        samples = []
        while len(samples) < sample_size:
            ''' generate a following vehicle '''
            fol_v = random.sample(cv, 1)[0]
            
            ''' in what frames this vehicle exists '''
            fol_frames = np.asarray(list(self.dict_by_veh[fol_v].keys()))
            
            ''' the intial frame '''
            ini_frame = fol_frames[0] 

            ''' the initial lane position '''
            ini_lane = self.dict_by_veh[fol_v][ini_frame]["lane"]
            
            if fol_frames.shape[0] < 300:
                continue
            
            ''' select a frame as the start point '''
            sta_frame = random.sample(set(fol_frames[:period]), 1)[0]
            end_frame = sta_frame + period * 100
            
            ''' search for vehicles the fol_veh was following through these frames '''
            cur_frame = sta_frame
            convoy = {}
            if_append = 1
            while cur_frame < end_frame:
                try:
                    fol_loc = self.dict_by_veh[fol_v][cur_frame]["loc"]
                    fol_lane = self.dict_by_veh[fol_v][cur_frame]["lane"]
                except KeyError:
                    if_append = 0
                    break
                ''' the following vehicle shouldn't have applied lane changing '''
                if ini_lane != fol_lane:
                    if_append = 0
                    break
                
                ''' put information in the convoy dictionary '''
                convoy[cur_frame] = {}    
                convoy[cur_frame][fol_v] = self.dict_by_veh[fol_v][cur_frame]
                convoy[cur_frame][fol_v]["is_fol"] = 1 
                
                ''' all vehicles in the lane at the current frame '''
                all_vehs = self.dict_by_frame[fol_lane][cur_frame] 
        
                for led_v in all_vehs.keys():
                    led_loc = self.dict_by_veh[led_v][cur_frame]["loc"]
                    if led_v not in cv:
                        continue
                    if not 0 < led_loc - fol_loc < convoy_len:
                        continue
                    convoy[cur_frame][led_v] = self.dict_by_veh[led_v][cur_frame]
                    convoy[cur_frame][led_v]["is_fol"] = 0
                    
                ''' move to the next frame '''    
                cur_frame += 100
            
            ''' append the new convoy sample to the sample set only if it is not empty '''
            if if_append:
                samples.append(convoy)
            print (str(len(samples)) + " convoy found!")
            
        self.samples = samples
        
        ''' save the samples by the current system time '''
        now = datetime.datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        path = './data/CF_' + str(dt_string)
        save_data(cf_samples, path)
        
        return samples
    
    ''' Import vital dictionaries from elsewhere '''
    def __Import_Dict__(self, dict_f_path = "./data/dictionary_frame", dict_v_path = "./data/dictionary_vehicle"):
        self.dict_by_frame = load_data(dict_f_path)
        self.dict_by_veh = load_data(dict_v_path)
            
def Display_CF(cf):
    veh_xt = {}
    for frame in cf:
        for veh in cf[frame]:
            x = cf[frame][veh]["loc"]
            t = frame
            if veh not in veh_xt.keys():
                veh_xt[veh] = {"x":[], "t":[]}
            veh_xt[veh]["x"].append(x)
            veh_xt[veh]["t"].append(t)
    for veh in veh_xt:
        plt.plot(veh_xt[veh]["t"], veh_xt[veh]["x"])
    plt.show()
    
class Car_Following():
    def __init__(self, cf_dict, model = "hu"):
        self.cf = cf_dict
        self.model = model
        self.fol_veh = None
        self.have_sim = False
        self.sim = {}
        self.collide = False
        
        ''' parameter to calibrate '''
        self.a = None
        self.tr = None
        
        ''' get the fol_veh id'''
        self.__get_fol_veh__()
        
        
    def __calibrate__(self, tr_range = [0, 3], a_range = [0, 3]):
        
        ''' initialize min_lse '''
        min_lse = 1e9
        X = np.arange(tr_range[0], tr_range[1], 0.1)
        Y = np.arange(a_range[0], a_range[1], 0.1)
        arr = np.zeros(shape = (X.shape[0], Y.shape[0]))
        
        for i,tr in enumerate(np.arange(tr_range[0], tr_range[1], 0.1)):
            for j,a in enumerate(np.arange(a_range[0], a_range[1], 0.1)):
                self.__simulate__(tr, a)
                # if self.collide:
                #     # print ("collide warning!")
                #     continue
                lse = self.__display__(if_plot = False)
                arr[i][j] = math.log(lse)
                if lse < min_lse:
                    min_lse = lse
                    opt_a = a
                    opt_tr = tr
                    print (min_lse, a, tr)
                    
        self.__simulate__(opt_tr, opt_a)
        opt_lse = self.__display__()
        
        self.a = opt_a
        self.tr = opt_tr
         
        print ("The optimal tr and a are: " + str(opt_tr) + ", " + str(opt_a) + " with LSE " + str(opt_lse))  
        
        # self.__plot__(X, Y, arr)
        
        return X, Y, arr
        
    def __calibrate_tr__(self, tr_range = [0.1, 3.0], a = 17.5):
        
        ''' initialize min_lse '''
        min_lse = 1e9
        for tr in np.arange(tr_range[0], tr_range[1], 0.1):
            
            self.__simulate__(tr, a)
            if self.collide:
                # print ("collide warning!")
                continue
            lse = self.__display__(if_plot = False)
            if lse < min_lse:
                min_lse = lse
                opt_a = a
                opt_tr = tr
                # print (min_lse, a, tr)
        self.__simulate__(opt_tr, opt_a)
        opt_lse = self.__display__()
        
        self.a = opt_a
        self.tr = opt_tr
        
        # print ("The optimal tr and a are: " + str(opt_tr) + ", " + str(opt_a) + " with LSE " + str(opt_lse))  
        
    def __SGD_calibrate__(self, learning_rate = 1e-1, momentum = 0.9):

        n = learning_rate
        r = momentum
        
        ''' initialize tr, a, and point estimator '''
        tr = 0.6
        a = 15.0
        optimal_a = 0
        optimal_tr = 0 
        min_lse = 1e9
        
        lse = 1e9
        prev_lse = 1e8 
        m_a = 0
 
        ''' until converge '''
        while abs(prev_lse - lse) > 1e-3:
            
            ''' attain the gradient on a '''
            self.__simulate__(tr, a-0.01)
            lse_minus = self.__display__(if_plot = False)
            self.__simulate__(tr, a+0.01)
            lse_plus = self.__display__(if_plot = False)
            grad_a = (lse_plus - lse_minus) / 0.02
            
            ''' update a '''
            prev_m_a = m_a
            m_a = m_a * r + (1 - r) * n * grad_a
            
            ''' update the previous estimator '''
            prev_lse = lse
            
            ''' attain the current estimator '''
            self.__simulate__(tr, a)
            lse = self.__display__(if_plot = False)
                
            ''' see the current lse to track the model '''
            # print (lse, a)
            
            ''' should parameters be updated? '''
            if prev_lse < lse:
                break
            
            optimal_a = a
            optimal_tr = tr
            
            a = a - m_a
               
        self.__simulate__(optimal_tr, optimal_a)
        lse = self.__display__()
        self.a = optimal_a
        self.tr = optimal_tr
        print ("The optimal tr and a are: " + str(optimal_tr) + ", " + str(optimal_a)  + " with LSE " + str(lse))
        
    def __SGD_calibrate_tr_a__(self, learning_rate = 1e-2, momentum = 0.9):

        n = learning_rate
        r = momentum
        
        ''' initialize tr, a, and point estimator '''
        tr_range = np.arange(0, 3, 0.1)
        optimal_a = 1.5
        optimal_tr = 0 
        min_lse = 1e9

        m_a = 0
        
        for tr in tr_range:
            ''' until converge '''
            prev_lse = 1e8 
            lse = 1e9
            
            ''' initilize a '''
            a = optimal_a
            
            while abs(prev_lse - lse) > 1e-3:
              
                
                ''' attain the gradient on a '''
                self.__simulate__(tr, a-0.01)
                lse_minus = self.__display__(if_plot = False)
                self.__simulate__(tr, a+0.01)
                lse_plus = self.__display__(if_plot = False)
                grad_a = (lse_plus - lse_minus) / 0.02
                
                if math.isnan(grad_a):
                    break
                
                ''' how much to update a '''
                prev_m_a = m_a
                m_a = m_a * r + (1 - r) * n * grad_a
                
                ''' update the previous estimator '''
                prev_lse = lse
                
                ''' attain the current estimator '''
                self.__simulate__(tr, a)
                lse = self.__display__(if_plot = False)
                
                ''' update the optimal '''
                if lse < min_lse:
                    min_lse = lse
                    optimal_a = a
                    optimal_tr = tr
                    # print (lse, a, tr)
                    
                ''' see the current lse to track the model '''
                
                
                ''' should parameters be updated? '''
                if prev_lse < lse:
                    break
                
                ''' update a '''
                a = a - m_a
                  
                ''' control the plausible range of a '''
                if not 0 < a <= 3:
                    break
                
                # print (lse, a, tr)
               
        self.__simulate__(optimal_tr, optimal_a)
        lse = self.__display__()
        self.a = optimal_a
        self.tr = optimal_tr
        # print ("The optimal tr and a are: " + str(optimal_tr) + ", " + str(optimal_a)  + " with LSE " + str(lse))
            
            
    def __3Dplot__(self, X, Y, arr):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        X, Y = np.meshgrid(X, Y)
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        surf = ax.plot_surface(X, Y, arr, cmap="winter_r", linewidth=0, antialiased=True)
        
        fig.colorbar(surf, shrink = 0.5, aspect = 5)
        now = datetime.datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        ax.set_ylabel("sensitivity")
        ax.set_xlabel("reaction time /s")
        ax.set_zlabel("ln(Chi-square)")
        plt.savefig("3D" + str(dt_string), dpi = 300)
        plt.show()
    
    def __contour_map__(self, X, Y, arr):
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import matplotlib.tri as tri
        X, Y = np.meshgrid(X, Y)
        fig, ax = plt.subplots(1, 1)
        ax.contour(X, Y, arr, levels=14, linewidths=0.5, colors='k')
        
        cntr1 = ax.contourf(X, Y, arr, cmap="autumn_r")
        fig.colorbar(cntr1, ax=ax)
        ax.set_ylabel("sensitivity")
        ax.set_xlabel("reaction time /s")
        # ax.imshow(arr)
        # ax.imshow(weights, extent=(xmin, xmax, ymin, ymax), cmap=cmap)
        now = datetime.datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        plt.savefig("contour" + str(dt_string), dpi = 300)
        plt.show()
    
    def __get_fol_veh__(self):
        ''' get the first frame of the sample '''
        ini_frame = np.min(np.asarray(list(self.cf.keys())))
        
        ''' it's a dumb way to find fol_veh id because there's a bug in the above scripts that I don't know what happens'''
        last_loc = 10e7
        for veh in self.cf[ini_frame]:
            loc = self.cf[ini_frame][veh]["loc"]
            if loc < last_loc:
                last_veh = veh
                last_loc = loc
        self.fol_veh = last_veh

        
    def __simulate__(self, tr = 1.0, a = 15.00, l = 2):
        ''' get the first frame of the sample '''
        ini_frame = np.min(np.asarray(list(self.cf.keys())))
        end_frame = np.max(np.asarray(list(self.cf.keys())))
        
        ''' get the first frame to be considered in the fol_veh's motion (ini_frame + tr * 100) ''' 
        fol_frame = ini_frame + 1000 * tr
        led_frame = ini_frame
        
        ''' get the initial condition of the fol_veh '''
        fol_vel = self.cf[fol_frame][self.fol_veh]["vel"]
        fol_loc = self.cf[fol_frame][self.fol_veh]["loc"]
        
        ''' initialize the trajectory and speeds of the fol_veh for the simulation '''
        fol_loc_history = [fol_loc]
        fol_vel_history = [fol_vel]
        fol_frame_history = [fol_frame]
        
        # ''' a condition to break iterations when collision occurs '''
        # if_collide = False 
        
        ''' let's loop through all frames of the sample '''
        while fol_frame + 100 < end_frame:
            
            ''' get all led_vehs' locations '''
            l_locs = []
            for veh in self.cf[led_frame]:
                led_loc = self.cf[led_frame][veh]["loc"]
                led_veh = self.cf[led_frame][veh]["veh"]
                l_locs.append(led_loc)
                
            l_locs = np.asarray(sorted(l_locs))
            
            # ''' check if collision have occurred '''
            # if np.partition(l_locs, 1)[1] < fol_loc:
            #     self.collide = True

            spacings = l_locs - fol_loc
            
            ''' get the weighted optimal speeds '''
            _s = 0
            for j, dx in enumerate(spacings):
                if j == 0:
                    continue
                elif j == 1:
                    _s += self.optimal_speed(dx / j) * 1 / l ** (j - 1)   
                   
                else:
                    _s += self.optimal_speed(dx / j) * (l - 1) / l ** j  
               
            w_opt_vel = _s

            
            ''' calculate the estimated acceleration of the fol_veh '''
            fol_acc = a * ( w_opt_vel - fol_vel )
            
            ''' update the velocity of the fol_veh in the next frame '''
            fol_vel = fol_vel + fol_acc * 0.1
            
            ''' predict the location of the fol_veh in the next frame '''
            fol_loc = max(fol_loc, fol_loc + fol_vel * 0.1 + 0.5 * fol_acc * 0.1 ** 2)
            
            ''' forward following frames and leading frames '''
            fol_frame += 100
            led_frame += 100
            
            ''' update the trajectory history '''
            fol_loc_history.append(fol_loc)
            fol_vel_history.append(fol_vel)
            fol_frame_history.append(fol_frame)
            
        ''' update the simulation status '''
        self.have_sim = True
        self.sim = {"x":fol_loc_history, "t":fol_frame_history, "v":fol_vel_history}
        self.collide = False
        return self.sim

    def optimal_speed(self, d_x, v1 = 6.75, v2 = 7.91, c1 = 0.13, c2 = 1.57, Lc = 5):
        
        ''' convert d_x in ft to m '''
        d_x /= 3.28084
        
        ''' get optimal speed in v/s '''
        opt_vel = v1 + v2 * math.tanh(c1* (d_x - Lc) - c2)
        
        ''' convert the optimal speed to ft/s '''
        opt_vel *= 3.28084
        
        return max(opt_vel, 0)
    

    
    def __display__(self, if_plot = True):
        if not self.have_sim:
            raise ("Please do simulation first! ")
            
        _dict = {}
        fol_loc_actual = []
        for frame in self.cf:
            for veh in self.cf[frame]:
                x = self.cf[frame][veh]["loc"]
                t = frame
                if veh not in _dict:
                    _dict[veh] = {"x":[], "t":[]}
                _dict[veh]["x"].append(x)
                _dict[veh]["t"].append(t)
                if veh == self.fol_veh and frame in self.sim["t"]:
                    fol_loc_actual.append(x)
        
        ''' plot the simulation result '''
        if if_plot:
            for veh in _dict:
                if veh == self.fol_veh:
                    plt.plot(_dict[veh]["t"], _dict[veh]["x"], color = "purple", linewidth = 0.5 )
                else:
                    plt.plot(_dict[veh]["t"], _dict[veh]["x"], color = "black", linewidth = 0.5 )
            
            plt.plot(self.sim["t"], self.sim["x"], color = "red", linestyle = "--")
            plt.show()
            
        ''' calculate the least square estimator '''
        sigma = 5.0
        fol_loc_actual = np.asarray(fol_loc_actual)
        fol_loc_est = np.asarray(self.sim["x"])
        lse = np.sum( ( (fol_loc_actual - fol_loc_est) / sigma ) **2 )
        return lse

''' generate random data '''
def supplement_data(a_list2, tr_list2, con_list, err = 0.1):
    import random
    curr_num = len(a_list2)
    need_num = 1000 - len(a_list2)
    a_l = list(a_list2)
    tr_l = list(tr_list2)
    n_l = list(np.arange(0, len(a_list2), 1))
    c_l = list(con_list)
    for i in range(need_num):
        n = random.sample(n_l, 1)[0]
        new_a = np.random.normal(a_list2[n], err)
        new_tr = tr_list2[n] + random.sample([-0.2, -0.1, 0, 0.1, 0.2], 1)[0]
        a_l.append(new_a)
        tr_l.append(new_tr)
        c_l.append(con_list[n])
    return np.asarray(a_l), np.asarray(tr_l), np.asarray(c_l)
    
''' give congestion status to each sample '''
def congestion_class(cf_samples):
    status = []
    for cf in cf_samples:
        spaces = []
        for frame in cf:
            locs = []
            for veh in cf[frame]:
                locs.append(cf[frame][veh]["loc"])
            locs = np.asarray(locs)
            _sp = np.max(locs) - np.min(locs)
            sp = _sp / (locs.shape[0] - 1)
            spaces.append(sp)
        spaces = np.asarray(spaces)

        if np.mean(spaces) > 55:
            status.append(0)
        else:
            status.append(1)
    status = np.asarray(status)  
    print (np.count_nonzero(status))
    return np.asarray(status)
    
def congestion_class_speed(cf_samples):
    status = []
    for cf in cf_samples:
        ov_Vels = []
        for frame in cf:
            vels = []
            for veh in cf[frame]:
                vels.append(cf[frame][veh]["vel"])
            vels = np.asarray(vels)
            
            ov_Vels.append(np.mean(vels))
        ov_Vels = np.asarray(ov_Vels)

        if np.mean(ov_Vels) > 25:
            status.append(0)
        else:
            status.append(1)
    status = np.asarray(status)  
    print (np.count_nonzero(status))
    return np.asarray(status)
     
    
def U_Test(sample_X, sample_Y):
    sample_Merge = np.append(sample_X, sample_Y)
    Labels = np.append(np.full(sample_X.shape[0],'X'), np.full(sample_Y.shape[0],'Y'))
    sort = sorted(zip(sample_Merge, Labels))
    tuples = zip(*sort)
    sample_Merge, Labels = np.asarray([list(tu) for tu in tuples])
    Nx = sample_X.shape[0]
    Ny = sample_Y.shape[0]
    Rx = np.sum(np.where(Labels == 'X')[0])
    Ry = np.sum(np.where(Labels == 'Y')[0])
    N = sample_X.shape[0]
    U = Nx*Ny + 1/2 *Nx *(Nx+1) - Rx
    mu = 1/2 *Nx * Ny
    s = math.sqrt( 1/12 *Nx *Ny *(Nx + Ny + 1) )
    z = (U - mu)/s
    print (Nx, Ny)
    return z, U, mu, s, Rx, Ry

if __name__ == "__main__":
    
    0
    us101 = Dataset()
    us101_data = us101.__Import_Data__()
    # dictionary_frame = us101.__Dict_by_Frame__()
    # us101.__Import_Dict__()
    # cf_samples = us101.__Sampling__()
    
    # save_data(cf_samples, "cf2")
    
    # cf_samples = load_data("./data/CF_04_12_2020_20_47_01")
    # cf_sample1 = cf_samples[39]
    # Display_CF(cf_sample1)
    
    # cf_sample2 = cf_samples[899]
    # Display_CF(cf_sample2)
    
    # CF = Car_Following(cf_sample2)
    
    # X, Y, arr = CF.__calibrate__()
    # Car_Following(cf_sample2).__3Dplot__(X, Y, arr)
    # Car_Following(cf_sample2).__contour_map__(X, Y, arr)
    ''' SGD calibrate the tr when a = 17.0 ''' 
    # tr_list2 = []
    # a_list2 = []
    
    # for i, cf in enumerate(cf_samples):
    #     print ("Progress: " + str(i) + "/1000")
    #     try:
    #         CF = Car_Following(cf)
    #         CF.__SGD_calibrate_tr_a__()

    #         tr_list2.append(CF.tr)
    #         a_list2.append(CF.a)
    #         print ("tr:" + str(CF.tr) +"," + "a:" + str(CF.a))
    #     except:
    #         print ("A bug occurs ...")
    #         continue
    
    # now = datetime.datetime.now()
    # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # path = './data/CF_' + str(dt_string)
    # save_data(tr_list2, "./data/a_distribution" + str(dt_string))
    # save_data(a_list2, "./data/a_distribution" + str(dt_string))
    
    # plt.hist(a_list2, color = "green" , edgecolor = 'black', bins = np.arange(0, 3, 0.05), density = True)
    # plt.scatter(tr_list2, a_list2, color = "black", s = 1)
    
    ''' plot distribution of tr and a '''
    
    # save_data(a_l, "./data/tr_distribution" + str(dt_string))
    # save_data(tr_l, "./data/a_distribution" + str(dt_string))
    
    # a_l = load_data( "./data/a_distribution06_12_2020_22_20_48" )
    # tr_l = load_data( "./data/tr_distribution06_12_2020_22_20_48")
    
    # a_l, tr_l, c_l = supplement_data(a_l[:150], tr_l[:150], cong_V[:150], err = 0.1)
    # plt.style.use("ggplot")
    # plt.hist(tr_l, color = "green", edgecolor = 'black', bins = np.arange(0, 3, 0.1), density = True)
    # plt.xlabel("reaction time /s")
    # plt.ylabel("frequency")
    # plt.savefig("reaction_Freq", dpi = 300)
    # plt.show()
    
    # plt.hist(a_l, color = "orange" , edgecolor = 'black', bins = np.arange(0, 3, 0.05), density = True)
    
    # plt.xlabel("sensitivity")
    # plt.ylabel("frequency")
    # plt.savefig("sens_Freq", dpi = 300)
    # plt.show()
    
    # plt.scatter(tr_l[np.where(c_l == 1)], a_l[np.where(c_l == 1)], color = "red", s = 1)
    # plt.scatter(tr_l[np.where(c_l == 0)], a_l[np.where(c_l == 0)], color = "blue", s = 1)
    
    ''' simple z -test '''
    # m_tr_1 = np.mean(tr_l[np.where(c_l == 1)])
    # s_tr_1 = np.std(tr_l[np.where(c_l == 1)])                     
    # m_tr_0 = np.mean(tr_l[np.where(c_l == 0)])
    # s_tr_0 = np.std(tr_l[np.where(c_l == 0)])
    
    # z_tr = ( m_tr_1 - m_tr_0 ) / math.sqrt( s_tr_1 ** 2 + s_tr_0 ** 2 )
    
    ''' simple z -test '''
    # m_a_1 = np.mean(a_l[np.where(c_l == 1)])
    # s_a_1 = np.std(a_l[np.where(c_l == 1)])                     
    # m_a_0 = np.mean(a_l[np.where(c_l == 0)])
    # s_a_0 = np.std(a_l[np.where(c_l == 0)])
    
    # z_a = ( m_a_1 - m_a_0 ) / math.sqrt( s_a_1 ** 2 + s_a_0 ** 2 )
    # plt.hist(a_l[np.where(c_l == 0)], color = "blue" , edgecolor = 'black', bins = np.arange(0, 3, 0.05), density = True)
    # plt.hist(a_l[np.where(c_l == 1)], color = "red" , edgecolor = 'black', bins = np.arange(0, 3, 0.05), density = True)
                      
    # plt.xlabel("reaction time /s")
    # plt.ylabel("sensitivity")
    # plt.savefig("congestion", dpi = 300)
    # plt.show()
    
    ''' attain the congestion class '''
    # cong = congestion_class(cf_samples)
    # cong_V = congestion_class_speed(cf_samples)
    
    ''' U_test '''
    # scores_tr = U_Test(tr_l[np.where(c_l == 0)], (tr_l[np.where(c_l == 1)]))
    
    ''' SGD calibrate the a when tr = 0.6 ''' 
    # a_list = []
    
    # for i, cf in enumerate(cf_samples):
    #     print ("Progress: " + str(i) + "/1000")
    #     try:
    #         CF = Car_Following(cf)
    #         CF.__SGD_calibrate__()

    #         a_list.append(CF.a)
    #         print ("a:" + str(CF.a) +"," + "tr:" + str(CF.tr))
    #     except:
    #         print ("A bug occurs ...")
    #         continue
         

    # now = datetime.datetime.now()
    # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # path = './data/CF_' + str(dt_string)
    # save_data(a_list, "./data/a_distribution" + str(dt_string))
    
    
    
    # plt.hist(tr_list, color = "green" , edgecolor = 'black', bins = np.arange(0, 3, 0.1), density = True)
    
    
    # sim = CF.__simulate__()
    # lse = CF.__display__()
    
    # trial_opt_v = Car_Following(cf_sample2).optimal_speed(43)
    
    # now = datetime.datetime.now()
    # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # path = './data/CF_' + str(dt_string)
    # save_data(cf_samples, path)
    # dict_frame2 = load_data("./data/dictionary_frame")
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ''' generate a 60s car-following sample '''
    
    # s = random.sample(set(us101.unix_time), 1)[0]
    # def generate_a_car_following_sample(seed, lane = 1, period = 600, convoy_len = 600):
    #     frame = seed
    #     sample = {}
    #     while frame < period * 100 + seed:
    #         sample[frame] = dict_frame2[lane][frame] 
    #         frame += 100
    #     return sample
    
    # car_following_sample = generate_a_car_following_sample(s)
        
        
    
    
    # print ( dict_frame2 == dictionary_frame )

    
    # unix_time = us101.unix_time
    # datetimes = us101.datetimes

# -*- coding: utf-8 -*-



