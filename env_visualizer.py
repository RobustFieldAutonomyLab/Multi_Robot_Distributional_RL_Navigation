import marinenav_env.envs.marinenav_env as marinenav_env
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import copy
import scipy.spatial
import gym
import json
import os

class EnvVisualizer:

    def __init__(self, 
                 seed:int=0, 
                 draw_envs:bool=False, # Mode 2: plot the envrionment
                 draw_traj:bool=False, # Mode 3: plot final trajectories given action sequences
                 video_plots:bool=False, # Mode 4: Generate plots for a video
                 plot_dist:bool=False, # If return distributions are needed (for IQN agent) in the video
                 plot_qvalues:bool=False, # If Q values are needed in the video
                 dpi:int=96, # Monitor DPI
                 ): 
        self.env = marinenav_env.MarineNavEnv2(seed)
        self.env.reset()
        self.fig = None # figure for visualization
        self.axis_graph = None # sub figure for the map
        self.robots_plot = []
        self.robots_last_pos = []
        self.robots_traj_plot = []
        self.LiDAR_beams_plot = []
        self.axis_title = None # sub figure for title
        self.axis_action = None # sub figure for action command and steer data
        self.axis_goal = None # sub figure for relative goal measurment
        self.axis_perception = None # sub figure for perception output
        self.axis_dvl = None # sub figure for DVL measurement
        self.axis_dist = [] # sub figure(s) for return distribution of actions
        self.axis_qvalues = None # subfigure for Q values of actions

        self.episode_actions = [] # action sequence load from episode data
        self.episode_actions_quantiles = None
        self.episode_actions_taus = None

        self.plot_dist = plot_dist # draw return distribution of actions
        self.plot_qvalues = plot_qvalues # draw Q values of actions
        self.draw_envs = draw_envs # draw only the envs
        self.draw_traj = draw_traj # draw only final trajectories
        self.video_plots = video_plots # draw video plots
        self.plots_save_dir = None # video plots save directory 
        self.dpi = dpi # monitor DPI
        self.agent_name = None # agent name
        self.agent = None # agent (IQN or DQN for plot data)
        
        self.configs = None # evaluation configs
        self.episodes = None # evaluation episodes to visualize 

    def init_visualize(self,
                       env_configs=None # used in Mode 2
                       ):
        
        # initialize subplot for the map, robot state and sensor measurments
        if self.draw_envs:
            # Mode 2: plot the envrionment
            if env_configs is None:
                self.fig, self.axis_graph = plt.subplots(1,1,figsize=(8,8))
            else:
                num = len(env_configs)
                if num % 3 == 0:
                    self.fig, self.axis_graphs = plt.subplots(int(num/3),3,figsize=(8*3,8*int(num/3))) 
                else:
                    self.fig, self.axis_graphs = plt.subplots(1,num,figsize=(8*num,8))
        elif self.draw_traj:
            if self.plot_dist:
                self.fig = plt.figure(figsize=(24,16))
                spec = self.fig.add_gridspec(5,6)

                self.axis_graph = self.fig.add_subplot(spec[:,:4])
                self.axis_perception = self.fig.add_subplot(spec[:2,4:])
                self.axis_dist.append(self.fig.add_subplot(spec[2:,4]))
                self.axis_dist.append(self.fig.add_subplot(spec[2:,5]))
            else:
                # Mode 3: plot final trajectories given action sequences
                self.fig, self.axis_graph = plt.subplots(figsize=(16,16))
        elif self.video_plots:
            # Mode 4: Generate 1080p video plots
            w = 1920
            h = 1080
            self.fig = plt.figure(figsize=(w/self.dpi,h/self.dpi),dpi=self.dpi)
            if self.agent_name == "adaptive_IQN":
                spec = self.fig.add_gridspec(5,6)
                
                # self.axis_title = self.fig.add_subplot(spec[0,:])
                # self.axis_title.text(-0.9,0.,"adaptive IQN performance",fontweight="bold",fontsize=45)

                self.axis_graph = self.fig.add_subplot(spec[:,:4])
                self.axis_graph.set_title("adaptive IQN performance",fontweight="bold",fontsize=30)

                self.axis_perception = self.fig.add_subplot(spec[0:2,4:])
                self.axis_dist.append(self.fig.add_subplot(spec[2:,4]))
                self.axis_dist.append(self.fig.add_subplot(spec[2:,5]))
            elif self.agent_name == "IQN":
                spec = self.fig.add_gridspec(5,12)
                
                # self.axis_title = self.fig.add_subplot(spec[0,:])
                # self.axis_title.text(-0.9,0.,"IQN performance",fontweight="bold",fontsize=45)

                self.axis_graph = self.fig.add_subplot(spec[:,:8])
                self.axis_graph.set_title("IQN performance",fontweight="bold",fontsize=30)

                self.axis_perception = self.fig.add_subplot(spec[0:2,8:])
                self.axis_dist.append(self.fig.add_subplot(spec[2:,9:11]))
            elif self.agent_name == "DQN":
                spec = self.fig.add_gridspec(5,12)
                
                self.axis_graph = self.fig.add_subplot(spec[:,:8])
                self.axis_graph.set_title("DQN performance",fontweight="bold",fontsize=30)

                self.axis_perception = self.fig.add_subplot(spec[0:2,8:])
                self.axis_qvalues = self.fig.add_subplot(spec[2:,9:11])
            else:
                name = ""
                if self.agent_name == "APF":
                    name = "Artificial Potential Field"
                elif self.agent_name == "RVO":
                    name = "Reciprocal Velocity Obstacle" 
                spec = self.fig.add_gridspec(5,3)

                self.axis_graph = self.fig.add_subplot(spec[:,:2])
                self.axis_graph.set_title(f"{name} performance",fontweight="bold",fontsize=30)

                self.axis_perception = self.fig.add_subplot(spec[:2,2])
                self.axis_action = self.fig.add_subplot(spec[2:,2])

            # self.axis_title.set_xlim([-1.0,1.0])
            # self.axis_title.set_ylim([-1.0,1.0])
            # self.axis_title.set_xticks([])
            # self.axis_title.set_yticks([])
            # self.axis_title.spines["left"].set_visible(False)
            # self.axis_title.spines["top"].set_visible(False)
            # self.axis_title.spines["right"].set_visible(False)
            # self.axis_title.spines["bottom"].set_visible(False)
        else:
            # Mode 1 (default): Display an episode
            self.fig = plt.figure(figsize=(32,16))
            spec = self.fig.add_gridspec(5,4)
            self.axis_graph = self.fig.add_subplot(spec[:,:2])
            # self.axis_goal = self.fig.add_subplot(spec[0,2])
            self.axis_perception = self.fig.add_subplot(spec[1:3,2])
            # self.axis_dvl = self.fig.add_subplot(spec[3:,2])
            # self.axis_observation = self.fig.add_subplot(spec[:,3])

            ### temp for ploting head figure ###
            # self.fig, self.axis_graph = plt.subplots(1,1,figsize=(16,16))
            # # self.fig, self.axis_perception = plt.subplots(1,1,figsize=(8,8))

        if self.draw_envs and env_configs is not None:
            for i,env_config in enumerate(env_configs):
                self.load_env_config(env_config)
                if len(env_configs) % 3 == 0:
                    self.plot_graph(self.axis_graphs[int(i/3),i%3])
                else:
                    self.plot_graph(self.axis_graphs[i])
        else:
            self.plot_graph(self.axis_graph)

    def plot_graph(self,axis):
        # plot current velocity in the map
        # if self.draw_envs:
        #     x_pos = list(np.linspace(0.0,self.env.width,100))
        #     y_pos = list(np.linspace(0.0,self.env.height,100))
        # else:
        #     x_pos = list(np.linspace(-2.5,self.env.width+2.5,110))
        #     y_pos = list(np.linspace(-2.5,self.env.height+2.5,110))

        x_pos = list(np.linspace(0.0,self.env.width,100))
        y_pos = list(np.linspace(0.0,self.env.height,100))

        pos_x = []
        pos_y = []
        arrow_x = []
        arrow_y = []
        speeds = np.zeros((len(x_pos),len(y_pos)))
        for m,x in enumerate(x_pos):
            for n,y in enumerate(y_pos):
                v = self.env.get_velocity(x,y)
                speed = np.clip(np.linalg.norm(v),0.1,10)
                pos_x.append(x)
                pos_y.append(y)
                arrow_x.append(v[0])
                arrow_y.append(v[1])
                speeds[n,m] = np.log(speed)


        cmap = cm.Blues(np.linspace(0,1,20))
        cmap = mpl.colors.ListedColormap(cmap[10:,:-1])

        axis.contourf(x_pos,y_pos,speeds,cmap=cmap)
        axis.quiver(pos_x, pos_y, arrow_x, arrow_y, width=0.001,scale_units='xy',scale=2.0)

        # if not self.draw_envs:
        #     # plot the evaluation boundary
        #     boundary = np.array([[0.0,0.0],
        #                         [self.env.width,0.0],
        #                         [self.env.width,self.env.height],
        #                         [0.0,self.env.height],
        #                         [0.0,0.0]])
        #     axis.plot(boundary[:,0],boundary[:,1],color = 'r',linestyle="-.",linewidth=3)

        # plot obstacles in the map
        l = True
        for obs in self.env.obstacles:
            if l:
                axis.add_patch(mpl.patches.Circle((obs.x,obs.y),radius=obs.r,color='m'))
                l = False
            else:
                axis.add_patch(mpl.patches.Circle((obs.x,obs.y),radius=obs.r,color='m'))

        axis.set_aspect('equal')
        # if self.draw_envs:
        #     axis.set_xlim([0.0,self.env.width])
        #     axis.set_ylim([0.0,self.env.height])
        # else:
        #     axis.set_xlim([-2.5,self.env.width+2.5])
        #     axis.set_ylim([-2.5,self.env.height+2.5])
        axis.set_xlim([0.0,self.env.width])
        axis.set_ylim([0.0,self.env.height])
        axis.set_xticks([])
        axis.set_yticks([])

        # plot start and goal state of each robot
        for idx,robot in enumerate(self.env.robots):
            if not self.draw_envs:
                axis.scatter(robot.start[0],robot.start[1],marker="o",color="yellow",s=200,zorder=6)
                axis.text(robot.start[0]-1,robot.start[1]+1,str(idx),color="yellow",fontsize=25,zorder=8)
            axis.scatter(robot.goal[0],robot.goal[1],marker="*",color="yellow",s=650,zorder=6)
            axis.text(robot.goal[0]-1,robot.goal[1]+1,str(idx),color="yellow",fontsize=25,zorder=8)
            self.robots_last_pos.append([])
            self.robots_traj_plot.append([])

        self.plot_robots(axis)
    
    def plot_robots(self,axis,traj_color=None):
        if not self.draw_envs:
            for robot_plot in self.robots_plot:
                robot_plot.remove()
            self.robots_plot.clear()

        robot_scale = 1.5
        for i,robot in enumerate(self.env.robots):
            if robot.deactivated:
                continue

            d = np.matrix([[0.5*robot_scale*robot.length],[0.5*robot_scale*robot.width]])
            rot = np.matrix([[np.cos(robot.theta),-np.sin(robot.theta)], \
                            [np.sin(robot.theta),np.cos(robot.theta)]])
            d_r = rot * d
            xy = (robot.x-d_r[0,0],robot.y-d_r[1,0])

            angle_d = robot.theta / np.pi * 180
            
            if self.draw_traj:
                robot.check_reach_goal()
                c = "lime" if robot.reach_goal else 'r'
            else:
                c = 'lime'
                # draw robot velocity (add initial length to avoid being hidden by the robot plot)
                robot_r = 0.5*np.linalg.norm(np.array([robot.length,robot.width]))
                init_len = robot_scale * robot_r + 0.1
                velocity_len = np.linalg.norm(robot.velocity)
                scaled_len = (velocity_len + init_len) / velocity_len
                self.robots_plot.append(axis.quiver(robot.x,robot.y,scaled_len*robot.velocity[0],scaled_len*robot.velocity[1], \
                                                    color="r",width=0.005,headlength=5,headwidth=3,scale_units='xy',scale=1))

            # draw robot
            self.robots_plot.append(axis.add_patch(mpl.patches.Rectangle(xy,robot_scale*robot.length, \
                                                              robot_scale*robot.width, color=c, \
                                                              angle=angle_d,zorder=7)))
            # if not self.draw_envs:
            #     # draw robot perception range
            #     self.robots_plot.append(axis.add_patch(mpl.patches.Circle((robot.x,robot.y), \
            #                                                     robot.perception.range, color=c,
            #                                                     alpha=0.2)))
            # robot id
            self.robots_plot.append(axis.text(robot.x-1,robot.y+1,str(i),color="yellow",fontsize=25,zorder=8))

            if not self.draw_envs:
                if self.robots_last_pos[i] != []:
                    h = axis.plot((self.robots_last_pos[i][0],robot.x),
                                (self.robots_last_pos[i][1],robot.y),
                                color='tab:orange' if traj_color is None else traj_color[i],
                                linewidth=3.0)
                    self.robots_traj_plot[i].append(h)
                
                self.robots_last_pos[i] = [robot.x, robot.y]

    def plot_action_and_steer_state(self,action):
        self.axis_action.clear()

        a,w = self.env.robots[0].actions[action]

        if self.video_plots:
            self.axis_action.text(1,3,"action",fontsize=25)
            self.axis_action.text(1,2,f"a: {a:.2f}",fontsize=20)
            self.axis_action.text(1,1,f"w: {w:.2f}",fontsize=20)
            self.axis_action.set_xlim([0,2.5])
            self.axis_action.set_ylim([0,4])
        else:
            x_pos = 0.15
            self.axis_action.text(x_pos,6,"Steer actions",fontweight="bold",fontsize=15)
            self.axis_action.text(x_pos,5,f"Acceleration (m/s^2): {a:.2f}",fontsize=15)
            self.axis_action.text(x_pos,4,f"Angular velocity (rad/s): {w:.2f}",fontsize=15)
            
            # robot steer state
            self.axis_action.text(x_pos,2,"Steer states",fontweight="bold",fontsize=15)
            self.axis_action.text(x_pos,1,f"Forward speed (m/s): {self.env.robot.speed:.2f}",fontsize=15)
            self.axis_action.text(x_pos,0,f"Orientation (rad): {self.env.robot.theta:.2f}",fontsize=15)
            self.axis_action.set_ylim([-1,7])

        self.axis_action.set_xticks([])
        self.axis_action.set_yticks([])
        self.axis_action.spines["left"].set_visible(False)
        self.axis_action.spines["top"].set_visible(False)
        self.axis_action.spines["right"].set_visible(False)
        self.axis_action.spines["bottom"].set_visible(False)

    def plot_measurements(self,robot_idx,R_matrix=None):
        self.axis_perception.clear()
        # self.axis_observation.clear()
        # self.axis_dvl.clear()
        # self.axis_goal.clear()

        rob = self.env.robots[robot_idx]

        # if rob.reach_goal:
        #     print(f"robot {robot_idx} reached goal, no measurements are available!")
        #     return

        legend_size = 12
        font_size = 15
        
        rob.perception_output(self.env.obstacles,self.env.robots)

        # plot detected objects in the robot frame (rotate x-axis by 90 degree (upward) in the plot)
        range_c = 'g' if rob.cooperative else 'r'
        self.axis_perception.add_patch(mpl.patches.Circle((0,0), \
                                       rob.perception.range, color=range_c, \
                                       alpha = 0.2))
        
        # plot self velocity (add initial length to avoid being hidden by the robot plot)
        robot_scale = 1.5
        robot_r = 0.5*np.linalg.norm(np.array([rob.length,rob.width]))
        init_len = robot_scale * robot_r
        velocity_len = np.linalg.norm(rob.velocity)
        scaled_len = (velocity_len + init_len) / velocity_len

        abs_velocity_r = rob.perception.observation["self"][2:]
        self.axis_perception.quiver(0.0,0.0,-scaled_len*abs_velocity_r[1],scaled_len*abs_velocity_r[0], \
                                   color='r',width=0.008,headlength=5,headwidth=3,scale_units='xy',scale=1)
        
        robot_c = 'g'
        self.axis_perception.add_patch(mpl.patches.Rectangle((-0.5*robot_scale*rob.width,-0.5*robot_scale*rob.length), \
                                       robot_scale*rob.width,robot_scale*rob.length, color=robot_c))

        x_pos = 0
        y_pos = 0
        relation_pos = [[0.0,0.0]]

        for i,obs in enumerate(rob.perception.observation["static"]):
            # rotate by 90 degree 
            self.axis_perception.add_patch(mpl.patches.Circle((-obs[1],obs[0]), \
                                           obs[2], color="m"))
            relation_pos.append([-obs[1],obs[0]])
            # include into observation info
            # self.axis_observation.text(x_pos,y_pos,f"position: ({obs[0]:.2f},{obs[1]:.2f}), radius: {obs[2]:.2f}")
            # y_pos += 1

        # self.axis_observation.text(x_pos,y_pos,"Static obstacles",fontweight="bold",fontsize=15)
        # y_pos += 2

        if rob.cooperative:
            # for i,obj_history in enumerate(rob.perception.observation["dynamic"].values()):
            for i,obj in enumerate(rob.perception.observation["dynamic"]):
                # plot the current position
                # pos = obj_history[-1][:2]

                # plot velocity (rotate by 90 degree)
                velocity_len = np.linalg.norm(rob.velocity)
                scaled_len = (velocity_len + init_len) / velocity_len
                self.axis_perception.quiver(-obj[1],obj[0],-scaled_len*obj[3],scaled_len*obj[2],color="r", \
                                            width=0.008,headlength=5,headwidth=3,scale_units='xy',scale=1)
                
                # plot position (rotate by 90 degree)
                self.axis_perception.add_patch(mpl.patches.Circle((-obj[1],obj[0]), \
                                            rob.detect_r, color="g"))
                relation_pos.append([-obj[1],obj[0]])
                
                # include history into observation info
                # self.axis_observation.text(x_pos,y_pos,f"position: ({obj[0]:.2f},{obj[1]:.2f}), velocity: ({obj[2]:.2f},{obj[3]:.2f})")
                # y_pos += 1
            
            # self.axis_observation.text(x_pos,y_pos,"Other Robots",fontweight="bold",fontsize=15)
            # y_pos += 2
        

        if R_matrix is not None:
            # plot relation matrix
            length = len(R_matrix)
            assert len(relation_pos) == length, "The number of objects do not match size of the relation matrix"
            for i in range(length):
                for j in range(length):
                    self.axis_perception.plot([relation_pos[i][0],relation_pos[j][0]], \
                                              [relation_pos[i][1],relation_pos[j][1]],
                                              linewidth=2*R_matrix[i][j],color='k',zorder=0)

        type = "cooperative" if rob.cooperative else "non-cooperative"
        # self.axis_observation.text(x_pos,y_pos,f"Showing the observation of robot {robot_idx} ({type})",fontweight="bold",fontsize=20)

        self.axis_perception.set_xlim([-rob.perception.range-1,rob.perception.range+1])
        self.axis_perception.set_ylim([-rob.perception.range-1,rob.perception.range+1])
        self.axis_perception.set_aspect('equal')
        self.axis_perception.set_title(f'Robot {robot_idx}',fontsize=25)

        self.axis_perception.set_xticks([])
        self.axis_perception.set_yticks([])
        self.axis_perception.spines["left"].set_visible(False)
        self.axis_perception.spines["top"].set_visible(False)
        self.axis_perception.spines["right"].set_visible(False)
        self.axis_perception.spines["bottom"].set_visible(False)

        # self.axis_observation.set_ylim([-1,y_pos+1])
        # self.axis_observation.set_xticks([])
        # self.axis_observation.set_yticks([])
        # self.axis_observation.spines["left"].set_visible(False)
        # self.axis_observation.spines["top"].set_visible(False)
        # self.axis_observation.spines["right"].set_visible(False)
        # self.axis_observation.spines["bottom"].set_visible(False)


        # # plot robot velocity in the robot frame (rotate x-axis by 90 degree (upward) in the plot)
        # h1 = self.axis_dvl.arrow(0.0,0.0,0.0,1.0, \
        #                color='k', \
        #                width = 0.02, \
        #                head_width = 0.08, \
        #                head_length = 0.12, \
        #                length_includes_head=True, \
        #                label='steering direction')
        # # rotate by 90 degree
        # h2 = self.axis_dvl.arrow(0.0,0.0,-abs_velocity_r[1],abs_velocity_r[0], \
        #                color='r',width=0.02, head_width = 0.08, \
        #                head_length = 0.12, length_includes_head=True, \
        #                label='velocity wrt seafloor')
        # x_range = np.max([2,np.abs(abs_velocity_r[1])])
        # y_range = np.max([2,np.abs(abs_velocity_r[0])])
        # mpl.rcParams["font.size"]=12
        # self.axis_dvl.set_xlim([-x_range,x_range])
        # self.axis_dvl.set_ylim([-1,y_range])
        # self.axis_dvl.set_aspect('equal')
        # self.axis_dvl.legend(handles=[h1,h2],loc='lower center',fontsize=legend_size)
        # self.axis_dvl.set_title('Velocity Measurement',fontsize=font_size)

        # self.axis_dvl.set_xticks([])
        # self.axis_dvl.set_yticks([])
        # self.axis_dvl.spines["left"].set_visible(False)
        # self.axis_dvl.spines["top"].set_visible(False)
        # self.axis_dvl.spines["right"].set_visible(False)
        # self.axis_dvl.spines["bottom"].set_visible(False)


        # # give goal position info in the robot frame
        # goal_r = rob.perception.observation["self"][:2]
        # x1 = 0.07
        # x2 = x1 + 0.13
        # self.axis_goal.text(x1,0.5,"Goal Position (Relative)",fontsize=font_size)
        # self.axis_goal.text(x2,0.25,f"({goal_r[0]:.2f}, {goal_r[1]:.2f})",fontsize=font_size)

        # self.axis_goal.set_xticks([])
        # self.axis_goal.set_yticks([])
        # self.axis_goal.spines["left"].set_visible(False)
        # self.axis_goal.spines["top"].set_visible(False)
        # self.axis_goal.spines["right"].set_visible(False)
        # self.axis_goal.spines["bottom"].set_visible(False)

    def plot_return_dist(self,action):
        for axis in self.axis_dist:
            axis.clear()
        
        dist_interval = 1
        mean_bar = 0.35
        idx = 0

        xlim = [np.inf,-np.inf]
        for idx, cvar in enumerate(action["cvars"]):
            ylabelright=[]

            quantiles = np.array(action["quantiles"][idx])

            q_means = np.mean(quantiles,axis=0)
            max_a = np.argmax(q_means)
            for i, a in enumerate(self.env.robots[0].actions):
                q_mean = q_means[i]
                # q_mean = np.mean(quantiles[:,i])

                ylabelright.append(
                    "\n".join([f"a: {a[0]:.2f}",f"w: {a[1]:.2f}"])
                )

                # ylabelright.append(f"mean: {q_mean:.2f}")
                
                self.axis_dist[idx].axhline(i*dist_interval, color="black", linewidth=2.0, zorder=0)
                self.axis_dist[idx].scatter(quantiles[:,i], i*np.ones(len(quantiles[:,i]))*dist_interval,color="g", marker="x",s=80,linewidth=3)
                self.axis_dist[idx].hlines(y=i*dist_interval, xmin=np.min(quantiles[:,i]), xmax=np.max(quantiles[:,i]),zorder=0)
                if i == max_a:
                    self.axis_dist[idx].vlines(q_mean, ymin=i*dist_interval-mean_bar, ymax=i*dist_interval+mean_bar,color="red",linewidth=5)
                else:
                    self.axis_dist[idx].vlines(q_mean, ymin=i*dist_interval-mean_bar, ymax=i*dist_interval+mean_bar,color="blue",linewidth=3)

            self.axis_dist[idx].tick_params(axis="x", labelsize=15)
            self.axis_dist[idx].set_ylim([-1.0,i+1])
            self.axis_dist[idx].set_yticks([])
            if idx == len(action["cvars"])-1:
                self.axis_dist[idx].set_yticks(range(0,i+1))
                self.axis_dist[idx].yaxis.tick_right()
                self.axis_dist[idx].set_yticklabels(labels=ylabelright,fontsize=15)
            
            if len(action["cvars"]) > 1:
                if idx == 0:
                    self.axis_dist[idx].set_title("adpative: "+r'$\phi$'+f" = {cvar:.2f}",fontsize=20)
                else:
                    self.axis_dist[idx].set_title(r'$\phi$'+f" = {cvar:.2f}",fontsize=20)
            else:
                self.axis_dist[idx].set_title(r'$\phi$'+f" = {cvar:.2f}",fontsize=20)
            
            xlim[0] = min(xlim[0],np.min(quantiles)-5)
            xlim[1] = max(xlim[1],np.max(quantiles)+5)

        for idx, cvar in enumerate(action["cvars"]):
            # self.axis_dist[idx].xaxis.set_ticks(np.arange(xlim[0],xlim[1]+1,(xlim[1]-xlim[0])/5))
            self.axis_dist[idx].set_xlim(xlim)

    def plot_action_values(self,action):
        self.axis_qvalues.clear()

        dist_interval = 1
        mean_bar = 0.35
        ylabelright=[]

        q_values = np.array(action["qvalues"])
        max_a = np.argmax(q_values)
        for i, a in enumerate(self.env.robots[0].actions):
            ylabelright.append(
                "\n".join([f"a: {a[0]:.2f}",f"w: {a[1]:.2f}"])
            )
            self.axis_qvalues.axhline(i*dist_interval, color="black", linewidth=2.0, zorder=0)
            if i == max_a:
                self.axis_qvalues.vlines(q_values[i], ymin=i*dist_interval-mean_bar, ymax=i*dist_interval+mean_bar,color="red",linewidth=8)
            else:
                self.axis_qvalues.vlines(q_values[i], ymin=i*dist_interval-mean_bar, ymax=i*dist_interval+mean_bar,color="blue",linewidth=5)
        
        self.axis_qvalues.set_title("action values",fontsize=20)
        self.axis_qvalues.tick_params(axis="x", labelsize=15)
        self.axis_qvalues.set_ylim([-1.0,i+1])
        self.axis_qvalues.set_yticks(range(0,i+1))
        self.axis_qvalues.yaxis.tick_right()
        self.axis_qvalues.set_yticklabels(labels=ylabelright,fontsize=15)
        self.axis_qvalues.set_xlim([np.min(q_values)-5,np.max(q_values)+5])

    def one_step(self,actions,robot_idx=0):
        assert len(actions) == len(self.env.robots), "Number of actions not equal number of robots!"
        for i,action in enumerate(actions):
            rob = self.env.robots[i]
            current_velocity = self.env.get_velocity(rob.x, rob.y)
            rob.update_state(action,current_velocity)

        self.plot_robots()
        self.plot_measurements(robot_idx)
        # if not self.plot_dist and not self.plot_qvalues:
        #     self.plot_action_and_steer_state(action["action"])
        
        if self.step % self.env.robots[0].N == 0:
            if self.plot_dist:
                self.plot_return_dist(action)
            elif self.plot_qvalues:
                self.plot_action_values(action)

        self.step += 1

    def init_animation(self):
        # plot initial robot position
        self.plot_robots()

        # plot initial measurments
        # self.plot_measurements() 

    def visualize_control(self,action_sequence,start_idx=0):
        # update robot state and make animation when executing action sequence
        actions = []

        # counter for updating distributions plot
        self.step = start_idx
        
        for i,a in enumerate(action_sequence):
            for _ in range(self.env.robots[0].N):
                # action = {}
                # action["action"] = a
                # if self.video_plots:
                #     if self.plot_dist:
                #         action["cvars"] = self.episode_actions_cvars[i]
                #         action["quantiles"] = self.episode_actions_quantiles[i]
                #         action["taus"] = self.episode_actions_taus[i]
                #     elif self.plot_qvalues:
                #         action["qvalues"] = self.episode_actions_values[i]
                
                actions.append(a)

        if self.video_plots:
            for i,action in enumerate(actions):
                self.one_step(action)
                self.fig.savefig(f"{self.plots_save_dir}/step_{self.step}.png",pad_inches=0.2,dpi=self.dpi)
        else:
            # self.animation = animation.FuncAnimation(self.fig, self.one_step,frames=actions, \
            #                                         init_func=self.init_animation,
            #                                         interval=10,repeat=False)
            for i,action in enumerate(actions):
                self.one_step(action)
            plt.show()

    def load_env_config(self,episode_dict):
        episode = copy.deepcopy(episode_dict)

        self.env.reset_with_eval_config(episode)

        if self.plot_dist:
            # load action cvars, quantiles and taus
            self.episode_actions_cvars = episode["robot"]["actions_cvars"]
            self.episode_actions_quantiles = episode["robot"]["actions_quantiles"]
            self.episode_actions_taus = episode["robot"]["actions_taus"]
        elif self.plot_qvalues:
            # load action values
            self.episode_actions_values = episode["robot"]["actions_values"]

    def load_env_config_from_eval_files(self,config_f,eval_f,eval_id,env_id):
        with open(config_f,"r") as f:
            episodes = json.load(f)
        episode = episodes[f"env_{env_id}"]
        eval_file = np.load(eval_f,allow_pickle=True)
        episode["robot"]["action_history"] = copy.deepcopy(eval_file["actions"][eval_id][env_id])
        self.load_env_config(episode)

    def load_env_config_from_json_file(self,filename):
        with open(filename,"r") as f:
            episode = json.load(f)
        self.load_env_config(episode)

    # def play_episode(self,start_idx=0):
    #     for plot in self.robot_traj_plot:
    #         plot[0].remove()
    #     self.robot_traj_plot.clear()

    #     current_v = self.env.get_velocity(self.env.start[0],self.env.start[1])
    #     self.env.robot.reset_state(self.env.start[0],self.env.start[1], current_velocity=current_v)

    #     self.init_visualize()

    #     self.visualize_control(self.episode_actions,start_idx)

    def load_eval_config_and_episode(self,config_file,eval_file):
        with open(config_file,"r") as f:
            self.configs = json.load(f)

        self.episodes = np.load(eval_file,allow_pickle=True)

    def play_eval_episode(self,eval_id,episode_id,colors,robot_ids=None):
        self.env.reset_with_eval_config(self.configs[episode_id])
        self.init_visualize()
        
        trajectories = self.episodes["trajectories"][eval_id][episode_id]
        
        self.play_episode(trajectories,colors,robot_ids)

    def play_episode(self,
                     trajectories,
                     colors,
                     robot_ids=None,
                     max_steps=None,
                     start_step=0):
        
        # sort robots according trajectory lengths
        all_robots = []
        for i,traj in enumerate(trajectories):
            plot_observation = False if robot_ids is None else i in robot_ids
            all_robots.append({"id":i,"traj_len":len(traj),"plot_observation":plot_observation})
        all_robots = sorted(all_robots, key=lambda x: x["traj_len"])
        all_robots[-1]["plot_observation"] = True

        if max_steps is None:
            max_steps = all_robots[-1]["traj_len"]-1

        robots = []
        for robot in all_robots:
            if robot["plot_observation"] is True:
                robots.append(robot)

        idx = 0
        current_robot_step = 0
        for i in range(max_steps):
            if i >= robots[idx]["traj_len"]:
                current_robot_step = 0
                idx += 1
            self.plot_robots(self.axis_graph,colors)
            self.plot_measurements(robots[idx]["id"])
            # action = [actions[j][i] for j in range(len(self.env.robots))]
            # self.env.step(action)

            for j,rob in enumerate(self.env.robots):
                if rob.deactivated:
                    continue
                rob.x = trajectories[j][i][0]
                rob.y = trajectories[j][i][1]
                rob.theta = trajectories[j][i][2]
                rob.speed = trajectories[j][i][3]
                rob.velocity = np.array(trajectories[j][i][4:])

            if self.video_plots:
                if current_robot_step % self.env.robots[0].N == 0:
                    action = self.action_data(robots[idx]["id"])
                if self.agent_name == "adaptive_IQN" or self.agent_name == "IQN":
                    self.plot_return_dist(action)
                elif self.agent_name == "DQN":
                    self.plot_action_values(action)
                elif self.agent_name == "APF" or self.agent_name == "RVO":
                    self.plot_action_and_steer_state(action)
                self.fig.savefig(f"{self.plots_save_dir}/step_{start_step+i}.png",dpi=self.dpi)
            else:
                plt.pause(0.01)

            for j,rob in enumerate(self.env.robots):
                if i == len(trajectories[j])-1:
                    rob.deactivated = True

            current_robot_step += 1

    def draw_dist_plot(self,
                       trajectories,
                       robot_id,
                       step_id,
                       colors
                       ):
        
        self.init_visualize()

        for i in range(step_id+1):
            self.plot_robots(self.axis_graph,traj_color=colors)
            for j,rob in enumerate(self.env.robots):
                if rob.deactivated:
                    continue
                rob.x = trajectories[j][i+1][0]
                rob.y = trajectories[j][i+1][1]
                rob.theta = trajectories[j][i+1][2]
                rob.speed = trajectories[j][i+1][3]
                rob.velocity = np.array(trajectories[j][i+1][4:])
                if i+1 == len(trajectories[j])-1:
                    rob.deactivated = True

        # plot observation
        self.plot_measurements(robot_id)
        
        action = self.action_data(robot_id)
        self.plot_return_dist(action)

        self.fig.savefig("IQN_dist_plot.png",bbox_inches="tight")

    def action_data(self, robot_id):
        rob = self.env.robots[robot_id]
        observation,_,_ = rob.perception_output(self.env.obstacles,self.env.robots)

        if self.agent_name == "adaptive_IQN":
            # compute total distribution and adaptive CVaR distribution
            a_cvar,quantiles_cvar,_,cvar = self.agent.act_adaptive(observation)
            a_greedy,quantiles_greedy,_ = self.agent.act(observation)

            action = dict(action=[a_cvar,a_greedy],
                          cvars=[cvar,1.0],
                          quantiles=[quantiles_cvar[0],quantiles_greedy[0]])
        elif self.agent_name == "IQN":
            a_greedy,quantiles_greedy,_ = self.agent.act(observation)

            action = dict(action=[a_greedy],
                          cvars=[1.0],
                          quantiles=[quantiles_greedy[0]])
        elif self.agent_name == "DQN":
            a,qvalues = self.agent.act_dqn(observation)

            action = dict(action=a,qvalues=qvalues[0])
        elif self.agent_name == "APF" or self.agent_name == "RVO":
            action = self.agent.act(observation)
        return action


    def draw_trajectory(self,
                        trajectories,
                        colors,
                        name=None,
                        ):
        # Used in Mode 3
        self.init_visualize()

        # select a robot that is active utill the end of the episode
        robot_id = 0
        max_length = 0
        for i,traj in enumerate(trajectories):
            print("rob: ",i," len: ",len(traj))
            if len(traj) > max_length:
                robot_id = i
                max_length = len(traj)
        
        print("\n")

        for i in range(len(trajectories[robot_id])-1):
            self.plot_robots(self.axis_graph,traj_color=colors)
            for j,rob in enumerate(self.env.robots):
                if rob.deactivated:
                    continue
                rob.x = trajectories[j][i+1][0]
                rob.y = trajectories[j][i+1][1]
                rob.theta = trajectories[j][i+1][2]
                rob.speed = trajectories[j][i+1][3]
                rob.velocity = np.array(trajectories[j][i+1][4:])
                if i+1 == len(trajectories[j])-1:
                    rob.deactivated = True

        # for robot_plot in self.robots_plot:
        #     robot_plot.remove()
        # self.robots_plot.clear()

        fig_name = "trajectory_test.png" if name is None else f"trajectory_{name}.png"
        self.fig.savefig(fig_name,bbox_inches="tight")

    def draw_video_plots(self,episode,save_dir,start_idx,agent):
        # Used in Mode 4
        self.agent = agent
        self.load_env_config(episode)
        self.plots_save_dir = save_dir
        self.play_episode(start_idx)
        return self.step

