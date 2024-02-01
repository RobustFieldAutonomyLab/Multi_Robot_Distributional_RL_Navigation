import numpy as np
import copy

class APF_agent:

    def __init__(self, a, w):
        self.k_att = 50.0 # attractive force constant
        self.k_rep = 500.0 # repulsive force constant
        self.k_v = 1.0 # velocity force constant
        self.m = 500 # robot weight (kg)
        self.d0 = 15.0 # obstacle distance threshold (m)
        self.n = 2 # power constant of repulsive force
        self.min_vel = 1.0 # if velocity is lower than the threshold, mandate acceleration 

        self.a = a # available linear acceleration (action 1)
        self.w = w # available angular velocity (action 2)

        self.GAMMA = 0.99

    def position_force(self,position,radius,goal):
        d_obs = np.linalg.norm(position) - radius - 0.8
        d_goal = np.linalg.norm(goal)

        # repulsive force component to move away from the obstacle 
        mag_1 = self.k_rep * ((1/d_obs)-(1/self.d0)) * (d_goal ** self.n)/(d_obs ** 2)
        dir_1 = -1.0 * position / np.linalg.norm(position)
        F_rep_1 = mag_1 * dir_1

        # repulsive force component to move towards the goal
        mag_2 = (self.n / 2) * self.k_rep * (((1/d_obs)-(1/self.d0))**2) * (d_goal ** (self.n-1))
        dir_2 = -1.0 * goal / d_goal
        F_rep_2 = mag_2 * dir_2

        return F_rep_1+F_rep_2
    
    def velocity_force(self,v_ao,position,radius):
        d_obs = np.linalg.norm(position) - radius - 0.8
        
        mag = -self.k_v * v_ao / d_obs
        dir = position / np.linalg.norm(position)
        
        F_rep = mag * dir

        return F_rep

    def act(self, observation):
        assert len(observation) == 39, "The state size does not equal 39"

        obs_array = np.array(observation)
        
        ego = obs_array[:4]
        static = obs_array[4:19]
        dynamic = obs_array[19:] 

        # compute attractive force
        F_att = self.k_att * ego[:2]

        # compute total repulsive force from sonar reflections
        F_rep = np.zeros(2)
        goal = ego[:2]
        velocity = ego[2:]

        # force from static obstacles
        for i in range(0,len(static),3):
            if np.abs(static[i]) < 1e-3 and np.abs(static[i+1]) < 1e-3:
                # padding
                continue

            F_rep += self.position_force(static[i:i+2],static[i+2],goal)

        # force from dynamic obstacles
        for i in range(0,len(dynamic),4):
            if np.abs(dynamic[i]) < 1e-3 and np.abs(dynamic[i+1]) < 1e-3:
                # padding
                continue

            e_ao = dynamic[i:i+2]/np.linalg.norm(dynamic[i:i+2])
            v_ao = np.dot(velocity-dynamic[i+2:i+4],e_ao)
            if v_ao >= 0.0:
                # agent is moving towards dynamic obstacles
                F_rep += self.position_force(dynamic[i:i+2],0.8,goal)
                F_rep += self.velocity_force(v_ao,dynamic[i:i+2],0.8)

        # select angular velocity action 
        F_total = F_att + F_rep
        V_angle = 0.0
        if np.linalg.norm(velocity) > 1e-03:
            V_angle = np.arctan2(velocity[1],velocity[0])
        F_angle = np.arctan2(F_total[1],F_total[0])

        diff_angle = F_angle - V_angle
        while diff_angle < -np.pi:
            diff_angle += 2 * np.pi
        while diff_angle >= np.pi:
            diff_angle -= 2 * np.pi

        w_idx = np.argmin(np.abs(self.w-diff_angle))
        
        # select linear acceleration action
        a_total = F_total / self.m
        V_dir = np.array([1.0,0.0])
        if np.linalg.norm(velocity) > 1e-03:
            V_dir = velocity / np.linalg.norm(velocity)
        a_proj = np.dot(a_total,V_dir)
 
        a = copy.deepcopy(self.a)
        if np.linalg.norm(velocity) < self.min_vel:
            # if the velocity is small, mandate acceleration
            a[a<=0.0] = -np.inf
        a_diff = a-a_proj
        a_idx = np.argmin(np.abs(a_diff))

        return a_idx * len(self.w) + w_idx

        
        
