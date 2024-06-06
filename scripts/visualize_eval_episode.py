import sys
sys.path.insert(0,"../")
import env_visualizer
import os

if __name__ == "__main__":
    dir = "your/training/evaluation/file/directory"

    eval_configs = os.path.join(dir,"eval_configs.json")
    evaluations = os.path.join(dir,"evaluations.npz")

    ev = env_visualizer.EnvVisualizer(seed=231)

    colors = ["r","lime","cyan","orange","tab:olive","white","chocolate"]

    eval_id = -1
    eval_episode = 0

    ev.load_eval_config_and_episode(eval_configs,evaluations)
    ev.play_eval_episode(eval_id,eval_episode,colors)