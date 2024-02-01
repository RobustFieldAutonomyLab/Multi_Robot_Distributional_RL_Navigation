import marinenav_env.envs.marinenav_env as marinenav_env
from policy.agent import Agent
import numpy as np
import copy
import scipy.spatial
import json
from datetime import datetime
import time as t_module
import os
import matplotlib.pyplot as plt
import APF
import sys
sys.path.insert(0,"./thirdparty")
import RVO

def evaluation(state, agent, eval_env, use_rl=True, use_iqn=True, act_adaptive=True, save_episode=False):
    """Evaluate performance of the agent
    """
    
    rob_num = len(eval_env.robots)


    rewards = [0.0]*rob_num
    times = [0.0]*rob_num
    energies = [0.0]*rob_num
    computation_times = []

    end_episode = False
    length = 0
    
    while not end_episode:
        # gather actions for robots from agents 
        action = []
        for i,rob in enumerate(eval_env.robots):
            if rob.deactivated:
                action.append(None)
                continue

            assert rob.cooperative, "Every robot must be cooperative!"
        
            start = t_module.time()
            if use_rl:
                if use_iqn:
                    if act_adaptive:
                        a,_,_,_ = agent.act_adaptive(state[i])
                    else:
                        a,_,_ = agent.act(state[i])
                else:
                    a,_ = agent.act_dqn(state[i])
            else:
                a = agent.act(state[i])
            end = t_module.time()
            computation_times.append(end-start)

            action.append(a)

        # execute actions in the training environment
        state, reward, done, info = eval_env.step(action)
        
        for i,rob in enumerate(eval_env.robots):
            if rob.deactivated:
                continue
            
            assert rob.cooperative, "Every robot must be cooperative!"
            
            rewards[i] += agent.GAMMA ** length * reward[i]

            times[i] += rob.dt * rob.N
            energies[i] += rob.compute_action_energy_cost(action[i])

            if rob.collision or rob.reach_goal:
                rob.deactivated = True

        end_episode = (length >= 360) or eval_env.check_all_deactivated()
        length += 1

    success = True if eval_env.check_all_reach_goal() else False
    # success = 0
    # for rob in eval_env.robots:
    #     if rob.reach_goal:
    #         success += 1

    # save time and energy data of robots that reach goal
    success_times = []
    success_energies = []
    for i,rob in enumerate(eval_env.robots):
        if rob.reach_goal:
            success_times.append(times[i])
            success_energies.append(energies[i])

    if save_episode:
        trajectories = []
        for rob in eval_env.robots:
            trajectories.append(copy.deepcopy(rob.trajectory))
        return success, rewards, computation_times, success_times, success_energies, trajectories
    else:
        return success, rewards, computation_times, success_times, success_energies

def exp_setup(envs,eval_schedule,i):
    observations = []

    for test_env in envs:
        test_env.num_cooperative = eval_schedule["num_cooperative"][i]
        test_env.num_non_cooperative = eval_schedule["num_non_cooperative"][i]
        test_env.num_cores = eval_schedule["num_cores"][i]
        test_env.num_obs = eval_schedule["num_obstacles"][i]
        test_env.min_start_goal_dis = eval_schedule["min_start_goal_dis"][i]

        # save eval config
        state,_,_ = test_env.reset()
        observations.append(state)

    return observations

def dashboard(eval_schedule,i):
    print("\n======== eval schedule ========")
    print("num of cooperative agents: ",eval_schedule["num_cooperative"][i])
    print("num of non-cooperative agents: ",eval_schedule["num_non_cooperative"][i])
    print("num of cores: ",eval_schedule["num_cores"][i])
    print("num of obstacles: ",eval_schedule["num_obstacles"][i])
    print("min start goal dis: ",eval_schedule["min_start_goal_dis"][i])
    print("======== eval schedule ========\n") 

def run_experiment(eval_schedules):
    agents = [adaptive_IQN_agent,IQN_agent,DQN_agent,APF_agent,RVO_agent]
    names = ["adaptive_IQN","IQN","DQN","APF","RVO"]
    envs = [test_env_0,test_env_1,test_env_2,test_env_3,test_env_4]
    evaluations = [evaluation,evaluation,evaluation,evaluation,evaluation]

    colors = ["b","g","r","tab:orange","m"]

    save_trajectory = True

    dt = datetime.now()
    timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S")

    robot_nums = []
    # all_test_rob_exp = []
    all_successes_exp = []
    all_rewards_exp = []
    all_success_times_exp = []
    all_success_energies_exp =[]
    if save_trajectory:
        all_trajectories_exp = []
        all_eval_configs_exp = []

    for idx,count in enumerate(eval_schedules["num_episodes"]):
        dashboard(eval_schedules,idx)

        robot_nums.append(eval_schedules["num_cooperative"][idx])
        # all_test_rob = [0]*len(agents)
        all_successes = [[] for _ in agents]
        all_rewards = [[] for _ in agents]
        all_computation_times = [[] for _ in agents]
        all_success_times = [[] for _ in agents]
        all_success_energies = [[] for _ in agents]
        if save_trajectory:
            all_trajectories = [[] for _ in agents]
            all_eval_configs = [[] for _ in agents]

        for i in range(count):
            print("Evaluating all agents on episode ",i)
            observations = exp_setup(envs,eval_schedules,idx)
            for j in range(len(agents)):
                agent = agents[j]
                env = envs[j]
                eval_func = evaluations[j]
                name = names[j]
                
                if save_trajectory:
                    all_eval_configs[j].append(env.episode_data())

                # obs = env.reset()
                obs = observations[j]
                if save_trajectory:
                    if name == "adaptive_IQN":
                        success, rewards, computation_times, success_times, success_energies, trajectories = eval_func(obs,agent,env,save_episode=True)
                    elif name == "IQN":
                        success, rewards, computation_times, success_times, success_energies, trajectories = eval_func(obs,agent,env,act_adaptive=False,save_episode=True)
                    elif name == "DQN":
                        success, rewards, computation_times, success_times, success_energies, trajectories = eval_func(obs,agent,env,use_iqn=False,save_episode=True)
                    elif name == "APF":
                        success, rewards, computation_times, success_times, success_energies, trajectories = eval_func(obs,agent,env,use_rl=False,save_episode=True)
                    elif name == "RVO":
                        success, rewards, computation_times, success_times, success_energies, trajectories = eval_func(obs,agent,env,use_rl=False,save_episode=True)
                    else:
                        raise RuntimeError("Agent not implemented!")
                else:
                    if name == "adaptive_IQN":
                        success, rewards, computation_times, success_times, success_energies = eval_func(obs,agent,env)
                    elif name == "IQN":
                        success, rewards, computation_times, success_times, success_energies = eval_func(obs,agent,env,act_adaptive=False)
                    elif name == "DQN":
                        success, rewards, computation_times, success_times, success_energies = eval_func(obs,agent,env,use_iqn=False)
                    elif name == "APF":
                        success, rewards, computation_times, success_times, success_energies = eval_func(obs,agent,env,use_rl=False)
                    elif name == "RVO":
                        success, rewards, computation_times, success_times, success_energies = eval_func(obs,agent,env,use_rl=False)
                    else:
                        raise RuntimeError("Agent not implemented!")

                all_successes[j].append(success)
                # all_test_rob[j] += eval_schedules["num_cooperative"][idx]
                # all_successes[j] += success
                all_rewards[j] += rewards
                all_computation_times[j] += computation_times
                all_success_times[j] += success_times
                all_success_energies[j] += success_energies
                if save_trajectory:
                    all_trajectories[j].append(copy.deepcopy(trajectories))
        
        for k,name in enumerate(names):
            s_rate = np.sum(all_successes[k])/len(all_successes[k])
            # s_rate = all_successes[k]/all_test_rob[k]
            avg_r = np.mean(all_rewards[k])
            avg_compute_t = np.mean(all_computation_times[k])
            avg_t = np.mean(all_success_times[k])
            avg_e = np.mean(all_success_energies[k])
            print(f"{name} | success rate: {s_rate:.2f} | avg_reward: {avg_r:.2f} | avg_compute_t: {avg_compute_t} | \
                  avg_t: {avg_t:.2f} | avg_e: {avg_e:.2f}")
        
        print("\n")

        # all_test_rob_exp.append(all_test_rob)
        all_successes_exp.append(all_successes)
        all_rewards_exp.append(all_rewards)
        all_success_times_exp.append(all_success_times)
        all_success_energies_exp.append(all_success_energies)
        if save_trajectory:
            all_trajectories_exp.append(copy.deepcopy(all_trajectories))
            all_eval_configs_exp.append(copy.deepcopy(all_eval_configs))

    # save data
    if save_trajectory:
        exp_data = dict(eval_schedules=eval_schedules,
                        names=names,
                        all_successes_exp=all_successes_exp,
                        all_rewards_exp=all_rewards_exp,
                        all_success_times_exp=all_success_times_exp,
                        all_success_energies_exp=all_success_energies_exp,
                        all_trajectories_exp=all_trajectories_exp,
                        all_eval_configs_exp=all_eval_configs_exp 
                    )
    else:
        exp_data = dict(eval_schedules=eval_schedules,
                        names=names,
                        all_successes_exp=all_successes_exp,
                        all_rewards_exp=all_rewards_exp,
                        all_success_times_exp=all_success_times_exp,
                        all_success_energies_exp=all_success_energies_exp,
                    )


    exp_dir = f"experiment_data/exp_data_{timestamp}"
    os.makedirs(exp_dir)

    filename = os.path.join(exp_dir,"exp_results.json")
    with open(filename,"w") as file:
        json.dump(exp_data,file)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    bar_width = 0.25
    interval_scale = 1.5
    set_label = [True]*len(names)
    for i,robot_num in enumerate(robot_nums):
         
        all_successes = all_successes_exp[i]
        all_success_times = all_success_times_exp[i]
        all_success_energies = all_success_energies_exp[i] 
        for j,pos in enumerate([-2*bar_width,-bar_width,0.0,bar_width,2*bar_width]):
            # bar plot for success rate
            s_rate = np.sum(all_successes[j])/len(all_successes[j])
            if set_label[j]:
                ax1.bar(interval_scale*i+pos,s_rate,0.8*bar_width,color=colors[j],label=names[j])
                set_label[j] = False
            else:
                ax1.bar(interval_scale*i+pos,s_rate,0.8*bar_width,color=colors[j])
            
            # box plot for time
            box = ax2.boxplot(all_success_times[j],positions=[interval_scale*i+pos],flierprops={'marker': '.','markersize': 1},patch_artist=True)
            for patch in box["boxes"]:
                patch.set_facecolor(colors[j])
            for line in box["medians"]:
                line.set_color("black")

            # box plot for energy
            box = ax3.boxplot(all_success_energies[j],positions=[interval_scale*i+pos],flierprops={'marker': '.','markersize': 1},patch_artist=True)
            for patch in box["boxes"]:
                patch.set_facecolor(colors[j])
            for line in box["medians"]:
                line.set_color("black")
    
    ax1.set_xticks(interval_scale*np.arange(len(robot_nums)))
    ax1.set_xticklabels(robot_nums)
    ax1.set_title("Success Rate")
    ax1.legend()

    ax2.set_xticks(interval_scale*np.arange(len(robot_nums)))
    ax2.set_xticklabels([str(robot_num) for robot_num in eval_schedules["num_cooperative"]])
    ax2.set_title("Time")

    ax3.set_xticks(interval_scale*np.arange(len(robot_nums)))
    ax3.set_xticklabels([str(robot_num) for robot_num in eval_schedules["num_cooperative"]])
    ax3.set_title("Energy")

    fig1.savefig(os.path.join(exp_dir,"success_rate.png"))
    fig2.savefig(os.path.join(exp_dir,"time.png"))
    fig3.savefig(os.path.join(exp_dir,"energy.png"))


if __name__ == "__main__":
    seed = 3 # PRNG seed for all testing envs

    ##### adaptive IQN #####
    test_env_0 = marinenav_env.MarineNavEnv2(seed)

    save_dir = "pretrained_models/IQN/seed_9"
    device = "cpu"

    adaptive_IQN_agent = Agent(cooperative=True,device=device)
    adaptive_IQN_agent.load_model(save_dir,"cooperative",device)
    ##### adaptive IQN #####

    ##### IQN #####
    test_env_1 = marinenav_env.MarineNavEnv2(seed)

    save_dir = "pretrained_models/IQN/seed_9"
    device = "cpu"

    IQN_agent = Agent(cooperative=True,device=device)
    IQN_agent.load_model(save_dir,"cooperative",device)
    ##### IQN #####

    ##### DQN #####
    test_env_2 = marinenav_env.MarineNavEnv2(seed)

    save_dir = "pretrained_models/DQN/seed_9"
    device = "cpu"

    DQN_agent = Agent(cooperative=True,device=device,use_iqn=False)
    DQN_agent.load_model(save_dir,"cooperative",device)
    ##### DQN #####

    ##### APF #####
    test_env_3 = marinenav_env.MarineNavEnv2(seed)

    APF_agent = APF.APF_agent(test_env_3.robots[0].a,test_env_3.robots[0].w)
    ##### APF #####

    ##### RVO #####
    test_env_4 = marinenav_env.MarineNavEnv2(seed)

    RVO_agent = RVO.RVO_agent(test_env_4.robots[0].a,test_env_4.robots[0].w,test_env_4.robots[0].max_speed)
    ##### RVO #####

    eval_schedules = dict(num_episodes=[100,100,100,100,100],
                          num_cooperative=[3,4,5,6,7],
                          num_non_cooperative=[0,0,0,0,0],
                          num_cores=[4,5,6,7,8],
                          num_obstacles=[4,5,6,7,8],
                          min_start_goal_dis=[40.0,40.0,40.0,40.0,40.0]
                         )

    run_experiment(eval_schedules)

