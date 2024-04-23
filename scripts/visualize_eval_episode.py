import sys
sys.path.insert(0,"../")
import env_visualizer

if __name__ == "__main__":
    dir = "your/training/evaluation/file/directory"

    eval_configs = "eval_configs.json"
    evaluations = "evaluations.npz"

    ev = env_visualizer.EnvVisualizer(seed=231)

    colors = ["r","lime","cyan","orange","tab:olive","white","chocolate"]

    eval_id = -1
    eval_episode = 55

    ev.load_eval_config_and_episode(dir+eval_configs,dir+evaluations)
    ev.play_eval_episode(eval_id,eval_episode,colors)