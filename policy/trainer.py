import json
import numpy as np
import os
import copy
import time

class Trainer():
    def __init__(self,
                 train_env,
                 eval_env,
                 eval_schedule,
                 non_cooperative_agent=None,
                 cooperative_agent=None,
                 UPDATE_EVERY=4,
                 learning_starts=2000,
                 target_update_interval=10000,
                 exploration_fraction=0.25,
                 initial_eps=0.6,
                 final_eps=0.05
                 ):
        
        self.train_env = train_env
        self.eval_env = eval_env
        self.cooperative_agent = cooperative_agent
        self.noncooperative_agent = non_cooperative_agent
        self.eval_config = []
        self.create_eval_configs(eval_schedule)

        self.UPDATE_EVERY = UPDATE_EVERY
        self.learning_starts = learning_starts
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.initial_eps = initial_eps
        self.final_eps = final_eps

        # Current time step
        self.current_timestep = 0

        # Learning time step (start counting after learning_starts time step)
        self.learning_timestep = 0

        # Evaluation data
        self.eval_timesteps = []
        self.eval_actions = []
        self.eval_trajectories = []
        self.eval_rewards = []
        self.eval_successes = []
        self.eval_times = []
        self.eval_energies = []
        self.eval_obs = []
        self.eval_objs = []

    def create_eval_configs(self,eval_schedule):
        self.eval_config.clear()

        count = 0
        for i,num_episode in enumerate(eval_schedule["num_episodes"]):
            for _ in range(num_episode): 
                self.eval_env.num_cooperative = eval_schedule["num_cooperative"][i]
                self.eval_env.num_non_cooperative = eval_schedule["num_non_cooperative"][i]
                self.eval_env.num_cores = eval_schedule["num_cores"][i]
                self.eval_env.num_obs = eval_schedule["num_obstacles"][i]
                self.eval_env.min_start_goal_dis = eval_schedule["min_start_goal_dis"][i]

                self.eval_env.reset()

                # save eval config
                self.eval_config.append(self.eval_env.episode_data())
                count += 1

    def save_eval_config(self,directory):
        file = os.path.join(directory,"eval_configs.json")
        with open(file, "w+") as f:
            json.dump(self.eval_config, f)

    def learn(self,
              total_timesteps,
              eval_freq,
              eval_log_path,
              verbose=True):
        
        states,_,_ = self.train_env.reset()

        # # Sample CVaR value from (0.0,1.0)
        # cvar = 1 - np.random.uniform(0.0, 1.0)

        # current episode 
        ep_rewards = np.zeros(len(self.train_env.robots))
        ep_deactivated_t = [-1]*len(self.train_env.robots)
        ep_length = 0
        ep_num = 0
        
        while self.current_timestep <= total_timesteps:
            
            # start_all = time.time()
            eps = self.linear_eps(total_timesteps)
            
            # gather actions for robots from agents 
            # start_1 = time.time()
            actions = []
            for i,rob in enumerate(self.train_env.robots):
                if rob.deactivated:
                    actions.append(None)
                    continue

                if rob.cooperative:
                    if self.cooperative_agent.use_iqn:
                        action,_,_ = self.cooperative_agent.act(states[i],eps)
                    else:
                        action,_ = self.cooperative_agent.act_dqn(states[i],eps)
                else:
                    if self.noncooperative_agent.use_iqn:
                        action,_,_ = self.noncooperative_agent.act(states[i],eps)
                    else:
                        action,_ = self.noncooperative_agent.act_dqn(states[i],eps)
                actions.append(action)
            # end_1 = time.time()
            # elapsed_time_1 = end_1 - start_1
            # if self.current_timestep % 100 == 0:
            #     print("Elapsed time 1: {:.6f} seconds".format(elapsed_time_1))

            # start_2 = time.time()
            # execute actions in the training environment
            next_states, rewards, dones, infos = self.train_env.step(actions)
            # end_2 = time.time()
            # elapsed_time_2 = end_2 - start_2
            # if self.current_timestep % 100 == 0:
            #     print("Elapsed time 2: {:.6f} seconds".format(elapsed_time_2))

            # save experience in replay memory
            for i,rob in enumerate(self.train_env.robots):
                if rob.deactivated:
                    continue

                if rob.cooperative:
                    ep_rewards[i] += self.cooperative_agent.GAMMA ** ep_length * rewards[i]
                    if self.cooperative_agent.training:
                        self.cooperative_agent.memory.add((states[i], actions[i], rewards[i], next_states[i], dones[i]))
                else:
                    ep_rewards[i] += self.noncooperative_agent.GAMMA ** ep_length * rewards[i]
                    if self.noncooperative_agent.training:
                        self.noncooperative_agent.memory.add((states[i], actions[i], rewards[i], next_states[i], dones[i]))
                
                if rob.collision or rob.reach_goal:
                    rob.deactivated = True
                    ep_deactivated_t[i] = ep_length

            end_episode = (ep_length >= 1000) or self.train_env.check_all_deactivated()
            
            # Learn, update and evaluate models after learning_starts time step 
            if self.current_timestep >= self.learning_starts:
                # start_3 = time.time()

                for agent in [self.cooperative_agent,self.noncooperative_agent]:
                    if agent is None:
                        continue

                    if not agent.training:
                        continue

                    # Learn every UPDATE_EVERY time steps.
                    if self.current_timestep % self.UPDATE_EVERY == 0:
                        # If enough samples are available in memory, get random subset and learn
                        if agent.memory.size() > agent.BATCH_SIZE:
                            agent.train()

                    # Update the target model every target_update_interval time steps
                    if self.current_timestep % self.target_update_interval == 0:
                        agent.soft_update()

                # end_3 = time.time()
                # elapsed_time_3 = end_3 - start_3
                # if self.current_timestep % 100 == 0:
                #     print("Elapsed time 3: {:.6f} seconds".format(elapsed_time_3))

                # Evaluate learning agents every eval_freq time steps
                if self.current_timestep == self.learning_starts or self.current_timestep % eval_freq == 0: 
                    self.evaluation()
                    self.save_evaluation(eval_log_path)

                    for agent in [self.cooperative_agent,self.noncooperative_agent]:
                        if agent is None:
                            continue
                        if not agent.training:
                            continue
                        # save the latest models
                        agent.save_latest_model(eval_log_path)

                # self.learning_timestep += 1

            if end_episode:
                ep_num += 1
                
                if verbose:
                    # print abstract info of learning process
                    print("======== Episode Info ========")
                    print("current ep_length: ",ep_length)
                    print("current ep_num: ",ep_num)
                    print("current exploration rate: ",eps)
                    print("current timesteps: ",self.current_timestep)
                    print("total timesteps: ",total_timesteps)
                    print("======== Episode Info ========\n")
                    print("======== Robots Info ========")
                    for i,rob in enumerate(self.train_env.robots):
                        info = infos[i]["state"]
                        if info == "deactivated after collision" or info == "deactivated after reaching goal":
                            print(f"Robot {i} ep reward: {ep_rewards[i]:.2f}, {info} at step {ep_deactivated_t[i]}")
                        else:
                            print(f"Robot {i} ep reward: {ep_rewards[i]:.2f}, {info}")
                    print("======== Robots Info ========\n") 

                states,_,_ = self.train_env.reset()
                # cvar = 1 - np.random.uniform(0.0, 1.0)

                ep_rewards = np.zeros(len(self.train_env.robots))
                ep_deactivated_t = [-1]*len(self.train_env.robots)
                ep_length = 0
            else:
                states = next_states
                ep_length += 1

            # end_all = time.time()
            # elapsed_time_all = end_all - start_all
            # if self.current_timestep % 100 == 0:
            #     print("one step elapsed time: {:.6f} seconds".format(elapsed_time_all))
            
            self.current_timestep += 1

    def linear_eps(self,total_timesteps):
        
        progress = self.current_timestep / total_timesteps
        if progress < self.exploration_fraction:
            r = progress / self.exploration_fraction
            return self.initial_eps + r * (self.final_eps - self.initial_eps)
        else:
            return self.final_eps

    def evaluation(self):
        """Evaluate performance of the agent
        Params
        ======
            eval_env (gym compatible env): evaluation environment
            eval_config: eval envs config file
        """
        actions_data = []
        trajectories_data = []
        rewards_data = []
        successes_data = []
        times_data = []
        energies_data = []
        obs_data = []
        objs_data = []
        
        for idx, config in enumerate(self.eval_config):
            print(f"Evaluating episode {idx}")
            state,_,_ = self.eval_env.reset_with_eval_config(config)
            obs = [[copy.deepcopy(rob.perception.observed_obs)] for rob in self.eval_env.robots]
            objs = [[copy.deepcopy(rob.perception.observed_objs)] for rob in self.eval_env.robots]
            
            rob_num = len(self.eval_env.robots)

            rewards = [0.0]*rob_num
            times = [0.0]*rob_num
            energies = [0.0]*rob_num
            end_episode = False
            length = 0
            
            while not end_episode:
                # gather actions for robots from agents 
                action = []
                for i,rob in enumerate(self.eval_env.robots):
                    if rob.deactivated:
                        action.append(None)
                        continue

                    if rob.cooperative:
                        if self.cooperative_agent.use_iqn:
                            a,_,_ = self.cooperative_agent.act(state[i])
                        else:
                            a,_ = self.cooperative_agent.act_dqn(state[i])
                    else:
                        if self.noncooperative_agent.use_iqn:
                            a,_,_ = self.noncooperative_agent.act(state[i])
                        else:
                            a,_ = self.noncooperative_agent.act_dqn(state[i])

                    action.append(a)

                # execute actions in the training environment
                state, reward, done, info = self.eval_env.step(action)
                
                for i,rob in enumerate(self.eval_env.robots):
                    if rob.deactivated:
                        continue
                    
                    if rob.cooperative:
                        rewards[i] += self.cooperative_agent.GAMMA ** length * reward[i]
                    else:
                        rewards[i] += self.noncooperative_agent.GAMMA ** length * reward[i]
                    times[i] += rob.dt * rob.N
                    energies[i] += rob.compute_action_energy_cost(action[i])
                    obs[i].append(copy.deepcopy(rob.perception.observed_obs))
                    objs[i].append(copy.deepcopy(rob.perception.observed_objs))

                    if rob.collision or rob.reach_goal:
                        rob.deactivated = True

                end_episode = (length >= 1000) or self.eval_env.check_any_collision() or self.eval_env.check_all_deactivated()
                length += 1

            actions = []
            trajectories = []
            for rob in self.eval_env.robots:
                actions.append(copy.deepcopy(rob.action_history))
                trajectories.append(copy.deepcopy(rob.trajectory))

            success = True if self.eval_env.check_all_reach_goal() else False

            actions_data.append(actions)
            trajectories_data.append(trajectories)
            rewards_data.append(np.mean(rewards))
            successes_data.append(success)
            times_data.append(np.mean(times))
            energies_data.append(np.mean(energies))
            obs_data.append(obs)
            objs_data.append(objs)
        
        avg_r = np.mean(rewards_data)
        success_rate = np.sum(successes_data)/len(successes_data)
        idx = np.where(np.array(successes_data) == 1)[0]
        avg_t = None if np.shape(idx)[0] == 0 else np.mean(np.array(times_data)[idx])
        avg_e = None if np.shape(idx)[0] == 0 else np.mean(np.array(energies_data)[idx])

        print(f"++++++++ Evaluation Info ++++++++")
        print(f"Avg cumulative reward: {avg_r:.2f}")
        print(f"Success rate: {success_rate:.2f}")
        if avg_t is not None:
            print(f"Avg time: {avg_t:.2f}")
            print(f"Avg energy: {avg_e:.2f}")
        print(f"++++++++ Evaluation Info ++++++++\n")

        self.eval_timesteps.append(self.current_timestep)
        self.eval_actions.append(actions_data)
        self.eval_trajectories.append(trajectories_data)
        self.eval_rewards.append(rewards_data)
        self.eval_successes.append(successes_data)
        self.eval_times.append(times_data)
        self.eval_energies.append(energies_data)
        self.eval_obs.append(obs_data)
        self.eval_objs.append(objs_data)

    def save_evaluation(self,eval_log_path):
        filename = "evaluations.npz"
        
        np.savez(
            os.path.join(eval_log_path,filename),
            timesteps=np.array(self.eval_timesteps,dtype=object),
            actions=np.array(self.eval_actions,dtype=object),
            trajectories=np.array(self.eval_trajectories,dtype=object),
            rewards=np.array(self.eval_rewards,dtype=object),
            successes=np.array(self.eval_successes,dtype=object),
            times=np.array(self.eval_times,dtype=object),
            energies=np.array(self.eval_energies,dtype=object),
            obs=np.array(self.eval_obs,dtype=object),
            objs=np.array(self.eval_objs,dtype=object)
        )