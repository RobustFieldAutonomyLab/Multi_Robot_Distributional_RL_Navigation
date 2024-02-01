# from math import ceil, floor, sqrt
import copy
import numpy as np

# from math import cos, sin, tan, atan2, asin

# from math import pi as PI

class RVO_agent:

    def __init__(self, a, w, max_vel):
        self.min_vel = 1.0 # if velocity is lower than the threshold, mandate acceleration
        self.max_vel = max_vel

        self.a = a # available linear acceleration (action 1)
        self.w = w # available angular velocity (action 2)

        self.rob_r = 2*0.8
        self.GAMMA = 0.99

    def distance(self,pose1,pose2):
        """ compute Euclidean distance for 2D """
        return np.sqrt((pose1[0]-pose2[0])**2+(pose1[1]-pose2[1])**2)+0.001

    def act(self, observation):
        assert len(observation) == 39, "The state size does not equal 39"

        obs_array = np.array(observation)
            
        ego = obs_array[:4]
        static = obs_array[4:19]
        dynamic = obs_array[19:]

        # desired velocity (towards goal)
        goal = ego[:2]
        v_desire = self.max_vel * goal / np.linalg.norm(goal)

        vA = [ego[2],ego[3]]
        pA = [0.0,0.0]

        RVO_BA_all = []

        # velocity obstacle from dynamic obstacles
        for i in range(0,len(dynamic),4):
            if np.abs(dynamic[i]) < 1e-3 and np.abs(dynamic[i+1]) < 1e-3:
                # padding
                continue

            vB = [dynamic[i+2],dynamic[i+3]]
            pB = [dynamic[i],dynamic[i+1]]

            # use RVO
            transl_vB_vA = [pA[0]+0.5*(vB[0]+vA[0]), pA[1]+0.5*(vB[1]+vA[1])]

            dist_BA = self.distance(pA, pB)
            theta_BA = np.arctan2(pB[1]-pA[1], pB[0]-pA[0])

            if 2*self.rob_r > dist_BA:
                dist_BA = 2*self.rob_r
            
            theta_BAort = np.arcsin(2*self.rob_r/dist_BA)
            theta_ort_left = theta_BA+theta_BAort
            bound_left = [np.cos(theta_ort_left), np.sin(theta_ort_left)]
            theta_ort_right = theta_BA-theta_BAort
            bound_right = [np.cos(theta_ort_right), np.sin(theta_ort_right)]

            RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, 2*self.rob_r]
            RVO_BA_all.append(RVO_BA)

        # velocity obstacle from static obstacles
        for i in range(0,len(static),3):
            if np.abs(static[i]) < 1e-3 and np.abs(static[i+1]) < 1e-3:
                # padding
                continue

            vB = [0.0, 0.0]
            pB = [static[i],static[i+1]]

            transl_vB_vA = [pA[0]+vB[0], pA[1]+vB[1]]
            dist_BA = self.distance(pA, pB)
            theta_BA = np.arctan2(pB[1]-pA[1], pB[0]-pA[0])

            # over-approximation of square to circular
            OVER_APPROX_C2S = 1.5
            rad = 2*static[i+2]*OVER_APPROX_C2S
            if (rad+self.rob_r) > dist_BA:
                dist_BA = rad+self.rob_r

            theta_BAort = np.arcsin((rad+self.rob_r)/dist_BA)
            theta_ort_left = theta_BA+theta_BAort
            bound_left = [np.cos(theta_ort_left), np.sin(theta_ort_left)]
            theta_ort_right = theta_BA-theta_BAort
            bound_right = [np.cos(theta_ort_right), np.sin(theta_ort_right)]
            RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, rad+self.rob_r]
            RVO_BA_all.append(RVO_BA)

        vA_post = self.intersect(pA, v_desire, RVO_BA_all)

        # select angular velocity action 
        vA_angle = 0.0
        vA_post_angle = 0.0
        if np.linalg.norm(np.array(vA)) > 1e-03:
            vA_angle = np.arctan2(vA[1],vA[0])
        if np.linalg.norm(np.array(vA_post)) > 1e-03:
            vA_post_angle = np.arctan2(vA_post[1],vA_post[0])

        diff_angle = vA_post_angle - vA_angle
        while diff_angle < -np.pi:
            diff_angle += 2 * np.pi
        while diff_angle >= np.pi:
            diff_angle -= 2 * np.pi

        w_idx = np.argmin(np.abs(self.w-diff_angle))

        # select linear acceleration action
        vA_dir = np.array([1.0,0.0])
        if np.linalg.norm(np.array(vA)) > 1e-03:
            vA_dir = np.array(vA) / np.linalg.norm(np.array(vA))
        vA_post_proj = np.dot(vA_dir,np.array(vA_post))

        a_proj = (vA_post_proj - np.linalg.norm(np.array(vA))) / 0.5
        a = copy.deepcopy(self.a)
        if np.linalg.norm(np.array(vA)) < self.min_vel:
            # if the velocity is small, mandate acceleration
            a[a<=0.0] = -np.inf
        a_diff = a-a_proj
        a_idx = np.argmin(np.abs(a_diff))

        return a_idx * len(self.w) + w_idx


    def intersect(self,pA, vA, RVO_BA_all):
        # print '----------------------------------------'
        # print 'Start intersection test'
        norm_v = self.distance(vA, [0, 0])
        suitable_V = []
        unsuitable_V = []
        for theta in np.arange(0, 2*np.pi, 0.1):
            for rad in np.arange(0.02, norm_v+0.02, norm_v/5.0):
                new_v = [rad*np.cos(theta), rad*np.sin(theta)]
                suit = True
                for RVO_BA in RVO_BA_all:
                    p_0 = RVO_BA[0]
                    left = RVO_BA[1]
                    right = RVO_BA[2]
                    dif = [new_v[0]+pA[0]-p_0[0], new_v[1]+pA[1]-p_0[1]]
                    theta_dif = np.arctan2(dif[1], dif[0])
                    theta_right = np.arctan2(right[1], right[0])
                    theta_left = np.arctan2(left[1], left[0])
                    if self.in_between(theta_right, theta_dif, theta_left):
                        suit = False
                        break
                if suit:
                    suitable_V.append(new_v)
                else:
                    unsuitable_V.append(new_v)                
        new_v = vA[:]
        suit = True
        for RVO_BA in RVO_BA_all:                
            p_0 = RVO_BA[0]
            left = RVO_BA[1]
            right = RVO_BA[2]
            dif = [new_v[0]+pA[0]-p_0[0], new_v[1]+pA[1]-p_0[1]]
            theta_dif = np.arctan2(dif[1], dif[0])
            theta_right = np.arctan2(right[1], right[0])
            theta_left = np.arctan2(left[1], left[0])
            if self.in_between(theta_right, theta_dif, theta_left):
                suit = False
                break
        if suit:
            suitable_V.append(new_v)
        else:
            unsuitable_V.append(new_v)
        #----------------------        
        if suitable_V:
            # print 'Suitable found'
            vA_post = min(suitable_V, key = lambda v: self.distance(v, vA))
            new_v = vA_post[:]
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dif = [new_v[0]+pA[0]-p_0[0], new_v[1]+pA[1]-p_0[1]]
                theta_dif = np.arctan2(dif[1], dif[0])
                theta_right = np.arctan2(right[1], right[0])
                theta_left = np.arctan2(left[1], left[0])
        else:
            # print 'Suitable not found'
            tc_V = dict()
            for unsuit_v in unsuitable_V:
                tc_V[tuple(unsuit_v)] = 0
                tc = []
                for RVO_BA in RVO_BA_all:
                    p_0 = RVO_BA[0]
                    left = RVO_BA[1]
                    right = RVO_BA[2]
                    dist = RVO_BA[3]
                    rad = RVO_BA[4]
                    dif = [unsuit_v[0]+pA[0]-p_0[0], unsuit_v[1]+pA[1]-p_0[1]]
                    theta_dif = np.arctan2(dif[1], dif[0])
                    theta_right = np.arctan2(right[1], right[0])
                    theta_left = np.arctan2(left[1], left[0])
                    if self.in_between(theta_right, theta_dif, theta_left):
                        small_theta = np.abs(theta_dif-0.5*(theta_left+theta_right))
                        if np.abs(dist*np.sin(small_theta)) >= rad:
                            rad = np.abs(dist*np.sin(small_theta))
                        big_theta = np.arcsin(np.abs(dist*np.sin(small_theta))/rad)
                        dist_tg = np.abs(dist*np.cos(small_theta))-np.abs(rad*np.cos(big_theta))
                        if dist_tg < 0:
                            dist_tg = 0                    
                        tc_v = dist_tg/self.distance(dif, [0,0])
                        tc.append(tc_v)
                tc_V[tuple(unsuit_v)] = min(tc)+0.001
            WT = 0.2
            vA_post = min(unsuitable_V, key = lambda v: ((WT/tc_V[tuple(v)])+self.distance(v, vA)))
        return vA_post 

    def in_between(self, theta_right, theta_dif, theta_left):
        if abs(theta_right - theta_left) <= np.pi:
            if theta_right <= theta_dif <= theta_left:
                return True
            else:
                return False
        else:
            if (theta_left <0) and (theta_right >0):
                theta_left += 2*np.pi
                if theta_dif < 0:
                    theta_dif += 2*np.pi
                if theta_right <= theta_dif <= theta_left:
                    return True
                else:
                    return False
            if (theta_left >0) and (theta_right <0):
                theta_right += 2*np.pi
                if theta_dif < 0:
                    theta_dif += 2*np.pi
                if theta_left <= theta_dif <= theta_right:
                    return True
                else:
                    return False
    
    