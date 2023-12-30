import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        # actions = TODO
        ##########
        # actions contains 4 last frames, thats why I choose the last predicted actions?
        actions = self.critic.qa_values(observation)
        action = np.argmax(actions[-1])
        ##########

        return action.squeeze()