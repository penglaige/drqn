"""
https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import os
import itertools
import numpy as np
import random
from collections import namedtuple
from utils.replay_buffer import *
from utils.schedules import *
from utils.minecraft_wrappers import ENV
from logger import Logger
import time
import pickle

import MalmoPython

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

#CUDA variables
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

# dqn_learner
class TESTAgent(object):
    """A model test agent."""

    def __init__(self,env,q_func,optimizer_spec,num_actions,modelFile,
        exploration=LinearSchedule(200000, 0.1),
        stopping_criterion=None,
        replay_buffer_size=1000000,
        batch_size=32,
        hidden_dim=512,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=10,
        img_h=84,
        img_w=84,
        img_c=1,
        target_update_freq=10000,
        double_dqn=False,
        dueling_dqn=False,
        ):
        """Run Deep Q-learning algorithm.
        You can specify your own convnet using q_func.
        All schedules are w.r.t. total number of steps taken in the environment.
        Parameters
        ----------
        env: gym.Env
            gym environment to train on.
        env_id: string
            gym environment id for model saving.
        q_func: function
            Model to use for computing the q function.
        optimizer_spec: OptimizerSpec
            Specifying the constructor and kwargs, as well as learning rate schedule
            for the optimizer
        exploration: rl_algs.deepq.utils.schedules.Schedule
            schedule for probability of chosing random action.
        stopping_criterion: (env, t) -> bool
            should return true when it's ok for the RL algorithm to stop.
            takes in env and the number of steps executed so far.
        replay_buffer_size: int
            How many memories to store in the replay buffer.
        batch_size: int
            How many transitions to sample each time experience is replayed.
        modelFile: string
            the name of the model
        gamma: float
            Discount Factor
        learning_starts: int
            After how many environment steps to start replaying experiences
        learning_freq: int
            How many steps of environment to take between every experience replay
        frame_history_len: int
            How many past frames to include as input to the model.
        target_update_freq: int
            How many experience replay rounds (not steps!) to perform between
            each update to the target Q network
        grad_norm_clipping: float or None
            If not None gradients' norms are clipped to this value.
        """
        self.logger = Logger('./logs')
        self.env = env
        self.q_func = q_func
        self.optimizer_spec = optimizer_spec
        self.exploration = exploration
        self.stopping_criterion = stopping_criterion
        self.num_actions = num_actions
        self.modelFile = modelFile
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.frame_history_len = frame_history_len
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn

        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.input_shape = (self.img_h, self.img_w, self.frame_history_len * self.img_c)
        self.in_channels = 1

        # open model file:
        try:
            self.model = torch.load(modelFile, map_location='cpu')
        except:
            print("Open model file Error!")

        # define Q target and Q
        self.Q = self.q_func(self.in_channels, self.num_actions, self.hidden_dim, self.frame_history_len).type(dtype)
        self.Q.load_state_dict(self.model)
        #create replay buffer
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.frame_history_len,self.img_w,self.img_h,"scale")

        ###### RUN SETTING ####
        self.t = 0
        self.num_param_updates = 0
        self.mean_episode_reward      = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        self.last_obs = None
        self.Log_Every_N_Steps = 1000
        self.Save_Model_Every_N_Steps = 10000
        #########################

    #Set the logger
    def to_np(self,x):
        return x.data.cpu().numpy() #???

    def run(self, agent_host):
        #run many times
        if self.last_obs is None:
            self.last_obs = self.env.reset()

        for t in itertools.count():
            ### 1. Check stopping criterion
            if self.stopping_criterion is not None and self.stopping_criterion(agent_host, t):
                if self.env.canset():
                    self.last_obs = self.env.reset()
                else:
                    break

            ### 2. Step the agent and store the transition
            # store last frame, returned idx used later
            #if self.last_obs is None:
                #self.last_obs = self.env.reset()

            last_stored_frame_idx = self.replay_buffer.store_frame(self.last_obs)

            #get observations to input to Q network (need to append prev frames)
            observations = self.replay_buffer.encode_recent_observation()
            # observation = (10, h, w)
            # return list, len=10, (batch,1,h,w) * 10

            #before learning starts, choose actions randomly


            obs = []
            for i in range(self.frame_history_len):
                temp_frame = np.expand_dims(observations[i], axis=0) # 1, h, w
                temp_frame = np.expand_dims(temp_frame, axis = 0) #1, 1, h, w
                obs.append(temp_frame)
            obs = np.array(obs) # 10, 1, 1, h, w
            obs = torch.from_numpy(obs).type(dtype) / 255.0

            with torch.no_grad():
                #q_value_all_actions = self.Q(Variable(obs)).cpu()
                ##q_value_all_actions = self.Q(Variable(obs))
                ##action = ((q_value_all_actions).data.max(1)[1])[0]
                self.Q.hidden = self.Q.init_hidden(1)
                action = self.Q(obs).max(1)[1].view(1,1)

            time.sleep(0.1)
            obs, reward, done = self.env.step(action)

            #clipping the reward, noted in nature paper
            reward = np.clip(reward, -1.0, 1.0)

            #store effect of action
            self.replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)

            #reset env if reached episode boundary
            if done:
                print("----------Episode %d end!-------------" %len(self.env.episode_rewards))
                if self.env.canset():
                    while True:
                        obs = self.env.reset()
                        if obs.any():
                            break
                        else:
                            continue
                else:
                    print("--------Episode number max-----------")
                    break

            #update last_obs
            self.last_obs = obs

            ### 4. Log progress
            episode_rewards = self.env.get_average_rewards_per_episode()
            #total scores for each episode
            episode_scores = self.env.get_episode_rewards()
            if len(episode_rewards) > 0:
                self.mean_episode_reward = np.mean(episode_rewards[-100:])
                self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)
            if t % self.Log_Every_N_Steps == 0 and len(episode_rewards) > 0:
                print("---------------------------------------")
                print('Timestep %d' %(t,))
                print("mean reward (100 episodes) %f" % self.mean_episode_reward)
                print("best mean reward %f" % self.best_mean_episode_reward)
                print("episodes_done %d"  % len(episode_rewards))
                #sys.stdout,flush()

                #=================== TensorBoard logging =============#
                # (1) Log the scalar values
                info = {
                    'num_episodes': len(episode_rewards),
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, t+1)

                if len(episode_rewards) > 0:
                    info = {
                        'last_episode_rewards': episode_rewards[-1],
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, t+1)

                if (self.best_mean_episode_reward != -float('inf')):
                    info = {
                        'mean_episode_reward_last_100': self.mean_episode_reward,
                        'best_mean_episode_reward': self.best_mean_episode_reward
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, t+1)

            if t % self.Save_Model_Every_N_Steps == 0:
                average_score = episode_rewards
                total_score = episode_scores
                total_apple = self.env.total_apple
                exploration = self.exploration.value(t)

                data = {"average_score":average_score,
                        "total_score":total_score,
                        "total_apple":total_apple}

                if not os.path.exists("perform_records"):
                    os.makedirs("perform_records")
                add_str = ''
                if (self.double_dqn):
                    add_str = 'double'
                if (self.dueling_dqn):
                    add_str = 'dueling'
                save_path = "perform_records/%s_%s_ep%d.pkl" %(str("test"), add_str, t)
                f = open(save_path,"wb")
                pickle.dump(data,f)
                f.close()
