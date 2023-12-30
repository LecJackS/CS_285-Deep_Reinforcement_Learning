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

    # Merge everything in one dict with list as values
    results = {k: [dic[k] for dic in all_dicts] for k in all_dicts[0]}

    for metric, values in results.items():
        print('>>', metric, values)
        # From string to float
        values = [float(v) for v in values]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('{}'.format(metric))
        ax1.plot(search_space, values)
        
        ax1.plot(search_space, values,
                 label="{:.3f} / {:.3f}".format(np.mean(values), np.std(values)),
                 marker="o")
        ax2.boxplot(values)
        ax1.title.set_text("n = {}".format(len(results["Eval_MaxReturn"])))
        ax1.legend(title="Mean / Std")

        logdir = logdirs[-1]
        plt.savefig(logdir + "/" + metric + '_.png')
        plt.show()


if __name__ == "__main__":

    args = ['--expert_policy_file', '/home/jack/CS_285-Deep_Reinforcement_Learning/hw1/cs285/policies/experts/Ant.pkl',
            '--expert_data',        '/home/jack/CS_285-Deep_Reinforcement_Learning/hw1/cs285/expert_data/expert_data_Ant-v2.pkl',
            '--env_name',           'Ant-v2',
            '--exp_name',           'bc_ant',
            '--ep_len',             '1000',
            '--eval_batch_size',    '5000',
            '--train_batch_size',   '100',
            '--num_agent_train_steps_per_iter', '1000',
            #'--do_dagger',
            '--n_iter', '1',
            '--no_gpu',
            '--video_log_freq', '-1'
            ]
    #grid_train_steps = ['1', '5', '10', '100', '1000', '5000', '50000']
    #grid_n_iter = ['2', '10', '20', '40']
    #seeds = [int(s) for s in np.linspace(0, 10000, 10)]
    #grid_n_layers = ['1','2','3','5','10', '30']
    # Run and save statistics for each param
    logdirs = []
    #for seed in seeds:
    #for ts in grid_train_steps:
    # for n_it in grid_n_iter:
    #for n_it in grid_n_iter:
    #for n_layers in grid_n_layers:
    num_samples = 30
    seeds = np.random.choice(10000, num_samples, replace=False)
    for n in range(num_samples):
        print("\n>>>>>>>>>> Starting experiment with n={} ...".format(n))
        args += ['--seed', str(seeds[n])]
        #args += ['--num_agent_train_steps_per_iter', str(ts)]
        #args += ['--n_iter', str(n_it)]
        #args += ['--n_layers', str(n_layers)]
        logdir = run_experiment(args)
        logdirs.append(logdir)

    # Create plots

    plot_results(logdirs, np.arange(num_samples))

