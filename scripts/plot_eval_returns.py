import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os

if __name__ == "__main__":
    first = True
    seeds = [9]
    eval_agents = ["IQN","DQN"]
    colors = ["b","tab:orange"]
    data_dirs = ["your/IQN/training/evaluation/file/directory",
                 "your/DQN/training/evaluation/file/directory"]

    fig, (ax_rewards,ax_success_rate) = plt.subplots(1,2,figsize=(16,6))
    
    IQN_final = []
    DQN_final = []

    for idx, eval_agent in enumerate(eval_agents):
        all_rewards = []
        all_success_rates = []
        print(f"===== Plotting {eval_agent} results =====")
        for seed in seeds:
            seed_dir = "seed_"+str(seed)
            eval_data = np.load(os.path.join(data_dirs[idx],seed_dir,"evaluations.npz")) # DQN

            timesteps = eval_data['timesteps']
            rewards = np.mean(eval_data['rewards'],axis=1)
            success_rates = []
            for i in range(len(eval_data['timesteps'])):
                successes = eval_data['successes'][i]
                success_rates.append(np.sum(successes)/len(successes))

            if eval_agent == "IQN":
                IQN_final.append(success_rates[-1])
            else:
                DQN_final.append(success_rates[-1])

            all_rewards.append(rewards)
            all_success_rates.append(success_rates)

        all_rewards_mean = np.mean(all_rewards,axis=0)
        all_rewards_std = np.std(all_rewards,axis=0)/np.sqrt(np.shape(all_rewards)[0])
        all_success_rates_mean = np.mean(all_success_rates,axis=0)
        all_success_rates_std = np.std(all_success_rates,axis=0)

        mpl.rcParams["font.size"]=25
        [x.set_linewidth(1.5) for x in ax_rewards.spines.values()]
        ax_rewards.tick_params(axis="x", labelsize=22)
        ax_rewards.tick_params(axis="y", labelsize=22)
        ax_rewards.plot(timesteps,all_rewards_mean,linewidth=3,label=eval_agent,c=colors[idx],zorder=5-idx)
        ax_rewards.fill_between(timesteps,all_rewards_mean+all_rewards_std,all_rewards_mean-all_rewards_std,alpha=0.2,color=colors[idx],zorder=5-idx)
        ax_rewards.xaxis.set_ticks(np.arange(0,eval_data['timesteps'][-1]+1,1000000))
        scale_x = 1e-5
        ticks_x = ticker.FuncFormatter(lambda x, pos:'{0:g}'.format(x*scale_x))
        ax_rewards.xaxis.set_major_formatter(ticks_x)
        ax_rewards.set_xlabel("Timestep(x10^5)",fontsize=25)
        ax_rewards.set_title("Cumulative Reward",fontsize=25,fontweight='bold')

        [x.set_linewidth(1.5) for x in ax_success_rate.spines.values()]
        ax_success_rate.tick_params(axis="x", labelsize=22)
        ax_success_rate.tick_params(axis="y", labelsize=22)
        ax_success_rate.plot(timesteps,all_success_rates_mean,linewidth=3,label=eval_agent,c=colors[idx],zorder=5-idx)
        ax_success_rate.fill_between(timesteps,all_success_rates_mean+all_success_rates_std,all_success_rates_mean-all_success_rates_std,alpha=0.2,color=colors[idx],zorder=5-idx)
        ax_success_rate.xaxis.set_ticks(np.arange(0,eval_data['timesteps'][-1]+1,1000000))
        scale_x = 1e-5
        ticks_x = ticker.FuncFormatter(lambda x, pos:'{0:g}'.format(x*scale_x))
        ax_success_rate.xaxis.set_major_formatter(ticks_x)
        ax_success_rate.set_xlabel("Timestep(x10^5)",fontsize=25)
        ax_success_rate.yaxis.set_ticks(np.arange(0,1.1,0.2))
        ax_success_rate.set_title("Success Rate",fontsize=25,fontweight='bold')
        ax_success_rate.legend(loc="lower right",bbox_to_anchor=(1, 0.35))

    print(f"IQN final success rates of all models: {IQN_final}")
    print(f"DQN final success rates of all models: {DQN_final}")

    fig.tight_layout()
    fig.savefig("learning_curves.png")
