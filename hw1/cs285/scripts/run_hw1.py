import os
import time

from cs285.infrastructure.rl_trainer import RL_Trainer
from cs285.agents.bc_agent import BCAgent
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy

import argparse

class BC_Trainer(object):

    def __init__(self, params):

        #######################
        ## AGENT PARAMS
        #######################

        agent_params = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'max_replay_buffer_size': params['max_replay_buffer_size'],
            }

        self.params = params
        self.params['agent_class'] = BCAgent ## HW1: you will modify this
        self.params['agent_params'] = agent_params

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params) ## HW1: you will modify this

        #######################
        ## LOAD EXPERT POLICY
        #######################

        print('Loading expert policy from...', self.params['expert_policy_file'])
        self.loaded_expert_policy = LoadedGaussianPolicy(self.params['expert_policy_file'])
        print('Done restoring expert policy...')

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            initial_expertdata=self.params['expert_data'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            relabel_with_expert=self.params['do_dagger'],
            expert_policy=self.loaded_expert_policy,
        )


def main(new_args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from
    parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from
    parser.add_argument('--env_name', '-env', type=str, help='choices: Ant-v2, Humanoid-v2, Walker-v2, HalfCheetah-v2, Hopper-v2', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int)

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1000)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=1000)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=64)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args(new_args)

    # convert args to dictionary
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if args.do_dagger:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q2_'
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        # Use this prefix when submitting. The auto-grader uses this prefix.
        logdir_prefix = 'q1_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    ## directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)


    ###################
    ### RUN TRAINING
    ###################

    trainer = BC_Trainer(params)
    trainer.run_training_loop()
    return logdir

if __name__ == "__main__":

    args = ['--expert_policy_file', '/home/jack/CS_285-Deep_Reinforcement_Learning/hw1/cs285/policies/experts/Ant.pkl',
            '--expert_data', '/home/jack/CS_285-Deep_Reinforcement_Learning/hw1/cs285/expert_data/expert_data_Ant-v2.pkl',
            '--env_name', 'Ant-v2',
            '--exp_name', 'bc_ant',
            '--ep_len', '5000',
            '--eval_batch_size', '5000',
            '--train_batch_size', '1000',
            '--num_agent_train_steps_per_iter', '100',
            '--do_dagger',
            '--n_iter', '50',
            ]

    args = ['--expert_policy_file', '/home/jack/CS_285-Deep_Reinforcement_Learning/hw1/cs285/policies/experts/Humanoid.pkl',
            '--expert_data', '/home/jack/CS_285-Deep_Reinforcement_Learning/hw1/cs285/expert_data/expert_data_Humanoid-v2.pkl',
            '--env_name', 'Humanoid-v2',
            '--exp_name', 'bc_humanoid',
            '--ep_len', '10',
            '--eval_batch_size', '50',
            '--train_batch_size', '10',
            '--num_agent_train_steps_per_iter', '10'
            ]
    main(args)
