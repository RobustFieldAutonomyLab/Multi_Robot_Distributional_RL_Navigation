import sys
sys.path.insert(0,"../")
import env_visualizer

if __name__ == "__main__":
    dir = "../pretrained_models/IQN/seed_9/"

    eval_configs = "eval_configs.json"
    evaluations = "evaluations.npz"

    ev = env_visualizer.EnvVisualizer(seed=231)

    colors = ["r","lime","cyan","orange","tab:olive","white","chocolate"]

    ev.load_eval_config_and_episode(dir+eval_configs,dir+evaluations)
    ev.play_eval_episode(-1,13,colors)