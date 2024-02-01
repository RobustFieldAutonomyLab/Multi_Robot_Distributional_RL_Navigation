import numpy as np
import copy
import heapq

class Perception:

    def __init__(self,cooperative:bool=False):
        # 2D LiDAR model with detection area as a sector
        self.range = 15.0 # range of beams (meter)
        self.angle = 2 * np.pi # detection angle range
        # self.len_obs_history = 5 # the window size of observation history
        self.max_obs_num = 5 # the maximum number of obstacles to be considered
        self.max_obj_num = 5 # the maximum number of dyanmic objects to be considered
        self.observation_format(cooperative)
        self.observed_obs = [] # indices of observed static obstacles
        self.observed_objs = [] # indiced of observed dynamic objects

    def observation_format(self,cooperative:bool=False):
        # format: {"self": [goal,velocity], 
        #          "static":[[obs_1.x,obs_1.y,obs_1.r],...,[obs_n.x,obs_n.y,obs_n.r]],
        #          "dynamic":[[robot_1.x,robot_1.y,robot_1.vx,robot_1.vy],...,[robot_n.x,robot_n.y,robot_n.vx,robot_n.vy]]
        if cooperative:
            self.observation = dict(self=[],static=[],dynamic=[])
        else:
            self.observation = dict(self=[],static=[])


class Robot:

    def __init__(self,cooperative:bool=False):
        
        # parameter initialization
        self.cooperative = cooperative # if the robot is cooperative or not
        self.dt = 0.05 # discretized time step (second)
        self.N = 10 # number of time step per action
        self.perception = Perception(cooperative)
        self.length = 1.0 
        self.width = 0.5
        self.r = 0.8 # collision range
        self.detect_r = 0.5*np.sqrt(self.length**2+self.width**2) # detection range
        self.goal_dis = 2.0 # max distance to goal considered as reached
        self.obs_dis = 5.0 # min distance to other objects that is considered safe     
        self.max_speed = 2.0
        self.a = np.array([-0.4,0.0,0.4]) # linear accelerations (m/s^2)
        self.w = np.array([-np.pi/6,0.0,np.pi/6]) # angular velocities (rad/s)
        self.compute_k() # cofficient of water resistance
        self.compute_actions() # list of actions

        self.x = None # x coordinate
        self.y = None # y coordinate
        self.theta = None # steering heading angle
        self.speed = None # steering foward speed
        self.velocity = None # velocity wrt sea floor
        
        self.start = None # start position
        self.goal = None # goal position
        self.collision = False
        self.reach_goal = False
        self.deactivated = False # deactivate the robot if it collides with any objects or reaches the goal 

        self.init_theta = 0.0 # theta at initial position
        self.init_speed = 0.0 # speed at initial position

        self.action_history = [] # history of action commands in one episode
        self.trajectory = [] # trajectory in one episode

    def compute_k(self):
        self.k = np.max(self.a)/self.max_speed
    
    def compute_actions(self):
        self.actions = [(acc,ang_v) for acc in self.a for ang_v in self.w]

    def compute_actions_dimension(self):
        return len(self.actions)

    def compute_dist_reward_scale(self):
        # scale the distance reward
        return 1 / (self.max_speed * self.N * self.dt)
    
    def compute_penalty_matrix(self):
        # scale the penalty value to [-1,0]
        scale_a = 1 / (np.max(self.a)*np.max(self.a))
        scale_w = 1 / (np.max(self.w)*np.max(self.w))
        p = -0.5 * np.matrix([[scale_a,0.0],[0.0,scale_w]])
        return p

    def compute_action_energy_cost(self,action):
        # scale the a and w to [0,1]
        a,w = self.actions[action]
        a /= np.max(self.a)
        w /= np.max(self.w)
        return np.abs(a) + np.abs(w)
    
    def dist_to_goal(self):
        return np.linalg.norm(self.goal - np.array([self.x,self.y]))

    def check_reach_goal(self):
        if self.dist_to_goal() <= self.goal_dis:
            self.reach_goal = True

    def reset_state(self,current_velocity=np.zeros(2)):
        # only called when resetting the environment
        self.action_history.clear()
        self.trajectory.clear()
        self.x = self.start[0]
        self.y = self.start[1]
        self.theta = self.init_theta 
        self.speed = self.init_speed
        self.update_velocity(current_velocity)
        self.trajectory.append([self.x,self.y,self.theta,self.speed,self.velocity[0],self.velocity[1]]) 

    def get_robot_transform(self):
        # compute transformation from world frame to robot frame
        R_wr = np.matrix([[np.cos(self.theta),-np.sin(self.theta)],[np.sin(self.theta),np.cos(self.theta)]])
        t_wr = np.matrix([[self.x],[self.y]])
        return R_wr, t_wr

    def get_steer_velocity(self):
        return self.speed * np.array([np.cos(self.theta), np.sin(self.theta)])

    def update_velocity(self,current_velocity=np.zeros(2)):
        steer_velocity = self.get_steer_velocity()
        self.velocity = steer_velocity + current_velocity

    def update_state(self,action,current_velocity):
        # update robot position in one time step
        self.update_velocity(current_velocity)
        dis = self.velocity * self.dt
        self.x += dis[0]
        self.y += dis[1]
        
        # update robot speed in one time step
        a,w = self.actions[action]
        
        # assume that water resistance force is proportion to the speed
        self.speed += (a-self.k*self.speed) * self.dt
        self.speed = np.clip(self.speed,0.0,self.max_speed)
        
        # update robot heading angle in one time step
        self.theta += w * self.dt

        # warp theta to [0,2*pi)
        while self.theta < 0.0:
            self.theta += 2 * np.pi
        while self.theta >= 2 * np.pi:
            self.theta -= 2 * np.pi

    def check_collision(self,obj_x,obj_y,obj_r):
        d = self.compute_distance(obj_x,obj_y,obj_r)
        if d <= 0.0:
            self.collision = True

    def compute_distance(self,x,y,r,in_robot_frame=False):
        if in_robot_frame:
            d = np.sqrt(x**2+y**2) - r - self.r
        else:
            d = np.sqrt((self.x-x)**2+(self.y-y)**2) - r - self.r
        return d

    def check_detection(self,obj_x,obj_y,obj_r):
        proj_pos = self.project_to_robot_frame(np.array([obj_x,obj_y]),False)
        
        if np.linalg.norm(proj_pos) > self.perception.range + obj_r:
            return False
        
        angle = np.arctan2(proj_pos[1],proj_pos[0])
        if angle < -0.5*self.perception.angle or angle > 0.5*self.perception.angle:
            return False
        
        return True
    
    def project_to_robot_frame(self,x,is_vector=True):
        assert isinstance(x,np.ndarray), "the input needs to be an numpy array"
        assert np.shape(x) == (2,)

        x_r = np.reshape(x,(2,1))

        R_wr, t_wr = self.get_robot_transform()

        R_rw = np.transpose(R_wr)
        t_rw = -R_rw * t_wr 

        if is_vector:
            x_r = R_rw * x_r
        else:
            x_r = R_rw * x_r + t_rw

        x_r.resize((2,))
        return np.array(x_r)            


    def perception_output(self,obstacles,robots,in_robot_frame=True):
        if self.deactivated:
            return None, self.collision, self.reach_goal
        
        self.perception.observation["static"].clear()
        if self.cooperative:
            self.perception.observation["dynamic"].clear()

        ##### self observation #####
        if in_robot_frame:
            # vehicle velocity wrt seafloor in self frame
            abs_velocity_r = self.project_to_robot_frame(self.velocity)

            # goal position in self frame
            goal_r = self.project_to_robot_frame(self.goal,False)

            self.perception.observation["self"] = list(np.concatenate((goal_r,abs_velocity_r)))
        else:
            self.perception.observation["self"] = [self.x,self.y,self.theta,self.speed,self.velocity[0],self.velocity[1],self.goal[0],self.goal[1]]


        self.perception.observed_obs.clear()
        if self.cooperative:
            self.perception.observed_objs.clear()

        self.check_reach_goal()

        # min_distance = np.inf

        ##### static obstacles observation #####
        for i,obs in enumerate(obstacles):
            if not self.check_detection(obs.x,obs.y,obs.r):
                continue

            self.perception.observed_obs.append(i)

            if not self.collision:
                self.check_collision(obs.x,obs.y,obs.r)

            if in_robot_frame:
                pos_r = self.project_to_robot_frame(np.array([obs.x,obs.y]),False)
                self.perception.observation["static"].append([pos_r[0],pos_r[1],obs.r])
            else:
                self.perception.observation["static"].append([obs.x,obs.y,obs.r])
            # min_distance = min(min_distance,np.linalg.norm(pos_r)-obs.r)

        if self.cooperative:
            ##### dynamic objects observation #####
            for j,robot in enumerate(robots):
                if robot is self:
                    continue
                if robot.deactivated:
                    # This robot is in the deactivate state, and abscent from the current map
                    continue
                if not self.check_detection(robot.x,robot.y,robot.detect_r):
                    continue

                self.perception.observed_objs.append(j)
                
                if not self.collision:
                    self.check_collision(robot.x,robot.y,robot.r)

                if in_robot_frame:
                    pos_r = self.project_to_robot_frame(np.array([robot.x,robot.y]),False)
                    v_r = self.project_to_robot_frame(robot.velocity)
                    new_obs = list(np.concatenate((pos_r,v_r)))
                    # if j in self.perception.observation["dynamic"].copy().keys():
                    #     self.perception.observation["dynamic"][j].append(new_obs)
                    #     while len(self.perception.observation["dynamic"][j]) > self.perception.len_obs_history:
                    #         del self.perception.observation["dynamic"][j][0]
                    # else:
                    #     self.perception.observation["dynamic"][j] = [new_obs]
                    self.perception.observation["dynamic"].append(new_obs)
                    # min_distance = min(min_distance,np.linalg.norm(pos_r)-robot.r)
                else:
                    self.perception.observation["dynamic"].append([robot.x,robot.y,robot.velocity[0],robot.velocity[1]])

        # if self.cooperative:
        #     for idx in self.perception.observation["dynamic"].copy().keys():
        #         if idx not in self.perception.observed_objs:
        #             # remove the observation history if the object is not observed in the current step
        #             del self.perception.observation["dynamic"][idx]

        self_state = copy.deepcopy(self.perception.observation["self"])
        static_observations = copy.deepcopy(heapq.nsmallest(self.perception.max_obs_num,
                                                      self.perception.observation["static"],
                                                      key=lambda obs:self.compute_distance(obs[0],obs[1],obs[2],True)))
        
        static_states = []
        for static in static_observations:
            static_states += static
        static_states += [0.0,0.0,0.0]*(self.perception.max_obs_num-len(static_observations))

        if self.cooperative:
            # # remove object indices 
            # dynamic_states = copy.deepcopy(list(self.perception.observation["dynamic"].values()))
            # if fixed_dynamic_size:
            #     dynamic_states_v = []
            #     for obs_history in dynamic_states:
            #         # convert the observation history into a vector [s_{t-k},s_{t-k+1},...,s_t]
            #         pad_len = max(self.perception.len_obs_history-len(obs_history),0)
            #         dynamic_states_vectors = [0.,0.,0.,0.] * pad_len
            #         for obs in obs_history:
            #             dynamic_states_vectors += obs
            #         dynamic_states_v.append(dynamic_states_vectors)
            #     return (self_state,static_states,dynamic_states_v), collision, self.reach_goal
            # else:
            #     idx_array = []
            #     for idx,obs_history in enumerate(dynamic_states):
            #         # pad the dynamic observation and save the indices of exact lastest element
            #         idx_array.append([idx,len(obs_history)-1])
            #         while len(obs_history) < self.perception.len_obs_history:
            #             obs_history.append([0.,0.,0.,0.])
            
            #     return (self_state,static_states,dynamic_states,idx_array), collision, self.reach_goal
            dynamic_observations = copy.deepcopy(heapq.nsmallest(self.perception.max_obj_num, 
                                                           self.perception.observation["dynamic"],
                                                           key=lambda obj:self.compute_distance(obj[0],obj[1],self.r,True)))
        else:
            dynamic_observations = [[0.0,0.0,0.0,0.0]]*self.perception.max_obj_num
            
        dynamic_states = []
        for dynamic in dynamic_observations:
            dynamic_states += dynamic
        dynamic_states += [0.0,0.0,0.0,0.0]*(self.perception.max_obj_num-len(dynamic_observations))

        return self_state+static_states+dynamic_states, self.collision, self.reach_goal


    # def perception_prediction(self,obs_states,robots_states=[]):
    #     # perception of predicted future state (robot frame)

    #     ##### self observation ##### 
    #     # vehicle velocity wrt seafloor in self frame
    #     abs_velocity_r = self.project_to_robot_frame(self.velocity)

    #     # goal position in self frame
    #     goal_r = self.project_to_robot_frame(self.goal,False)

    #     self_state = list(np.concatenate((goal_r,abs_velocity_r)))

    #     collision = False
    #     self.check_reach_goal()

    #     ##### static obstatcles observation #####
    #     static_states = []
    #     for obs in obs_states:
    #         if not self.check_detection(obs[0],obs[1],obs[2]):
    #             continue

    #         if not collision:
    #             collision = self.check_collision(obs[0],obs[1],obs[2])

    #         pos_r = self.project_to_robot_frame(np.array([obs[0],obs[1]]),False)
    #         static_states.append([pos_r[0],pos_r[1],obs[2]])

    #     if self.cooperative:
    #         # dynamic objects observation in self frame
    #         dynamic_states = []
    #         for obj in robots_states:
    #             if not self.check_detection(obj[0],obj[1],self.detect_r):
    #                 continue

    #             if not collision:
    #                 collision = self.check_collision(obj[0],obj[1],self.r)

    #             pos_r = self.project_to_robot_frame(np.array([obj[0],obj[1]]),False)
    #             v_r = self.project_to_robot_frame(np.array([obj[2],obj[3]]))
    #             dynamic_states.append(list(np.concatenate((pos_r,v_r))))
            
    #         return (self_state,static_states,dynamic_states), collision, self.reach_goal
    #     else:
    #         return (self_state,static_states), collision, self.reach_goal

            




            

            

                     



