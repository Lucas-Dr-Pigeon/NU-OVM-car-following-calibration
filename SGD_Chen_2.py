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
        
        
    def __calibrate__(self, tr_range = [0.1, 3.0], a_range = [2, 20], b_range = [0, 50]):
        
        ''' initialize min_lse '''
        min_lse = 1e9
        for tr in np.arange(tr_range[0], tr_range[1], 0.1):
            for a in np.arange(a_range[0], a_range[1], 1):
                for b in np.arange(b_range[0], b_range[1], 1): 
                    self.__simulate__(tr, a, b)
                    if self.collide:
                        # print ("collide warning!")
                        continue
                    lse = self.__display__(if_plot = False)
                    if lse < min_lse:
                        min_lse = lse
                        opt_a = a
                        opt_b = b
                        opt_tr = tr
                        print (min_lse, a, b, tr)
        self.__simulate__(opt_tr, opt_a, opt_b)
        opt_lse = self.__display__()
        
        self.a = opt_a
        self.tr = opt_tr
        
        # print ("The optimal tr and a are: " + str(opt_tr) + ", " + str(opt_a) + " with LSE " + str(opt_lse))  
        
    def __SGD_calibrate__(self, learning_rate = 1e-2, momentum = 0.9):

        n = learning_rate
        r = momentum
        
        ''' initialize tr, a, and point estimator '''
        tr = 0.6
        a = 5.0
        b = 0.002
        optimal_a = 0
        optimal_b = 0
        optimal_tr = 0 
        min_lse = 1e9
        
        lse = 1e9
        prev_lse = 1e8 
        m_a = 0
        m_b = 0
 
        ''' until converge '''
        while abs(prev_lse - lse) > 1e-3:
            
            ''' attain the gradient on a '''
            self.__simulate__(tr, a-0.01, b)
            lse_minus = self.__display__(if_plot = False)
            self.__simulate__(tr, a+0.01, b)
            lse_plus = self.__display__(if_plot = False)
            grad_a = (lse_plus - lse_minus) / 0.02
            
            ''' update a '''
            prev_m_a = m_a
            m_a = m_a * r + (1 - r) * n * grad_a
            
            ''' attain the gradient on b '''
            self.__simulate__(tr, a, b-0.01)
            lse_minus = self.__display__(if_plot = False)
            self.__simulate__(tr, a, b+0.01)
            lse_plus = self.__display__(if_plot = False)
            grad_b = (lse_plus - lse_minus) / 0.02
            
            ''' update b '''
            prev_m_b = m_b
            m_b = m_b * r + (1 - r) * n * grad_b
            
            ''' update the previous estimator '''
            prev_lse = lse
            
            ''' attain the current estimator '''
            self.__simulate__(tr, a, b)
            lse = self.__display__(if_plot = False)
                
            ''' see the current lse to track the model '''
            print (lse)
            
            ''' should parameters be updated? '''
            if prev_lse < lse:
                break
            
            optimal_a = a
            optimal_b = b
            optimal_tr = tr
            
            a = a - m_a
            b = b - m_b
               
        self.__simulate__(optimal_tr, optimal_a, optimal_b)
        lse = self.__display__()
        self.a = optimal_a
        self.tr = optimal_tr
        self.tr = optimal_b
        print ("The optimal tr and a are: " + str(optimal_tr) + ", " + str(optimal_a) + ", " + str(optimal_b) + " with LSE " + str(lse))
            
            
    def __plot__(self, tr = 0.4):
        lse_list = []
        for a in np.arange(16, 24, 0.1):
            self.__simulate__(tr, a)
            lse = self.__display__(if_plot = False)
            lse_list.append(lse)
        plt.plot(np.arange(16, 24, 0.1), lse_list, color = "red")
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

        
    def __simulate__(self, tr = 1.0, a = 15.00, b = 18.00, l = 2):
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
            
            ''' check if collision have occurred '''
            if np.partition(l_locs, 1)[1] < fol_loc:
                self.collide = True

            spacings = l_locs - fol_loc
            
            ''' get the weighted optimal speeds '''
            _s = 0
            _x = 0
            for j, dx in enumerate(spacings):
                if j == 0:
                    continue
                elif j == 1:
                    _s += self.optimal_speed(dx / j) * 1 / l ** (j - 1)   
                    _x += dx / j * 1 / l ** (j - 1)  
                else:
                    _s += self.optimal_speed(dx / j) * (l - 1) / l ** j  
                    _x += dx / j * (l - 1) / l ** j  
            w_opt_vel = _s
            w_opt_x = _x
            des_x = self.desired_space(fol_vel)
            
            ''' calculate the estimated acceleration of the fol_veh '''
            fol_acc = a * ( w_opt_vel - fol_vel )- b * (w_opt_x - des_x)
            
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
    
    def desired_space(self, v, s0 = 15, T = 2.5):
        
        return v * s0 + T
    
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
    
    
if __name__ == "__main__":
    
    0
    # us101 = Dataset()
    # us101_data = us101.__Import_Data__()
    # dictionary_frame = us101.__Dict_by_Frame__()
    # us101.__Import_Dict__()
    # cf_samples = us101.__Sampling__()
    
    # save_data(cf_samples, "cf2")
    
    cf_samples = load_data("./data/CF_04_12_2020_20_47_01")
    # cf_sample1 = cf_samples[39]
    # Display_CF(cf_sample1)
    
    cf_sample2 = cf_samples[59]
    # Display_CF(cf_sample2)
    
    CF = Car_Following(cf_sample2)
    
    CF.__SGD_calibrate__()
    
     
    # a_list = []
    # tr_list = []
    
    # for i, cf in enumerate(cf_samples):
    #     print ("Progress: " + str(i) + "/1000")
    #     try:
    #         CF = Car_Following(cf)
    #         CF.__SGD_calibrate__()
    #         a_list.append(CF.a)
    #         tr_list.append(CF.tr)
    #         print ("a:" + str(CF.a) +"," + "tr:" + str(CF.tr))
    #     except:
    #         print ("A bug occurs ...")
    #         continue
         

    # now = datetime.datetime.now()
    # dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    # path = './data/CF_' + str(dt_string)
    # save_data(a_list, "./data/a_distribution" + str(dt_string))
    # save_data(tr_list, "./data/tr_distribution" + str(dt_string))
    
    
    # plt.hist(tr_list, color = "green" , edgecolor = 'black', bins = np.arange(0, 3, 0.2), density = True)
    
    
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

