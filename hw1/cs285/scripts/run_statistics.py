from run_hw1 import main as run_experiment
from os import listdir
from os.path import isfile, join
import json
import matplotlib.pyplot as plt
import numpy as np
# class stats():

def plot_results(logdirs, search_space):
    all_dicts = []
    for logdir in logdirs:
        for f in listdir(logdir):
            if str(f)[:7] == "metrics":
                # Metrics file
                with open(join(logdir, f)) as json_file:
                    data = dict(json.load(json_file))
                    all_dicts.append(data)

    # Merge everthing in one dict with list as values
    results = {k: [dic[k] for dic in all_dicts] for k in all_dicts[0]}
    for metric, values in results.items():
        print('>>', metric, values)
        # From string to float
        values = [float(v) for v in values]
        plt.plot(search_space, values)
        plt.plot(search_space, values,
                 label="{:.4f}/{:.4f}".format(np.mean(values), np.std(values)),
                 marker="o")
        plt.title(metric)
        plt.legend(title="Mean/Std")
        logdir = logdirs[-1]
        plt.savefig(logdir + "/" + metric + '_.png')
        plt.show()


if __name__ == "__main__":

    args = ['--expert_policy_file', '/home/jack/homework_fall2020/hw1/cs285/policies/experts/Ant.pkl',
            '--expert_data',        '/home/jack/homework_fall2020/hw1/cs285/expert_data/expert_data_Ant-v2.pkl',
            '--env_name',           'Ant-v2',
            '--exp_name',           'bc_ant',
            '--ep_len',             '500',
            '--eval_batch_size',    '5000',
            '--train_batch_size',   '1000',
            #'--num_agent_train_steps_per_iter', '100',
            '--do_dagger',
            '--n_iter', '10',
            ]
    grid_train_steps = ['1', '5', '10', '100', '1000', '5000', '50000']
    seeds = [int(s) for s in np.linspace(0, 10000, 10)]
    # Run and save statistics for each param
    logdirs = []
    #for seed in seeds:
    for ts in grid_train_steps:
        print("\n>>>>>>>>>> Starting experiment with ts={} ...".format(ts))
        #args += ['--seed', str(seed)]
        args += ['--num_agent_train_steps_per_iter', str(ts)]
        logdir = run_experiment(args)
        logdirs.append(logdir)

    # Create plots
    plot_results(logdirs, grid_train_steps)

