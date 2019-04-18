from __future__ import print_function
from __future__ import division
# Start the training environment
# Find, break, use tools, hdrln
# Find : item-apple, tool-axe,shears
# Break: item-block, tool-axe,shears
# Use:   item-sheep, tool-axe,shears
# hdrln: item-apple, block, sheep, tool-axe,shears
import torch
import torch.optim as optim
import argparse

from task_xml import apple_missionXML
from model import Qnetwork,DQN, Dueling_DQN
from learn import DRQNAgent, OptimizerSpec
from dqn_learn import DQNAgent
from random_player import Ramdom
from test_model import TESTAgent
from utils.schedules import *
from utils.minecraft_wrappers import ENV

from builtins import range
from past.utils import old_div
import MalmoPython
import os
import sys
import time
import json
import random
import logging
import struct
import socket
import malmoutils

# Global Variables
BATCH_SIZE = 32
HIDDEN_DIM = 512
REPLAY_BUFFER_SIZE = 200000
FRAME_HISTORY_LEN = 10
TARGET_UPDATE_FREQ = 2000
GAMMA = 0.99
LEARNING_FREQ = 4
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01
EXPLORATION_SCHEDULE = LinearSchedule(200000, 0.1)
LEARNING_STARTS = 30000

RESIZE_MODE   = 'scale'
RESIZE_WIDTH  = 84
RESIZE_HEIGHT = 84

CUDA_VISIBLE_DEVICES = 0

num_actions = 3

def stopping_criterion(agent_host, t):
    world_state = agent_host.getWorldState()
    return not world_state.is_mission_running

optimizer = OptimizerSpec(
    constructor=optim.RMSprop,
    kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)
)


#----------- Minecraft environment setting -----------------------------------------------------------------


#----------- Minecraft environment setting -----------------------------------------------------------------
malmoutils.fix_print()

agent_host = MalmoPython.AgentHost()
malmoutils.parse_command_line(agent_host)
recordingsDirectory = malmoutils.get_recordings_directory(agent_host)
train, test_model, gpu, dqn, double_dqn, dueling_dqn, random_play, modelFile = malmoutils.get_options(agent_host)

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

missionXML = apple_missionXML

validate = True

agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
agent_host.setRewardsPolicy(MalmoPython.RewardsPolicy.KEEP_ALL_REWARDS )

if not train:
    num_reps = 100
    # Global Variables
    BATCH_SIZE = 32
    HIDDEN_DIM = 512
    REPLAY_BUFFER_SIZE = 1000000
    FRAME_HISTORY_LEN = 10
    TARGET_UPDATE_FREQ = 500
    GAMMA = 0.99
    LEARNING_FREQ = 4
    LEARNING_RATE = 0.00025
    ALPHA = 0.95
    EPS = 0.01
    EXPLORATION_SCHEDULE = LinearSchedule(2000, 0.1)
    LEARNING_STARTS = 500
else:
    num_reps = 5000

print("num_reps:",num_reps)

my_mission_record = MalmoPython.MissionRecordSpec()
if recordingsDirectory:
    my_mission_record.recordRewards()
    my_mission_record.recordObservations()
    my_mission_record.recordCommands()
    if agent_host.receivedArgument("record_video"):
        my_mission_record.recordMP4(24,2000000)

env = ENV(agent_host, missionXML, validate, my_mission_record, logger, recordingsDirectory, MAX_EPISODE=num_reps)

# ------------Command Parser-------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("current available gpu numbers: %d" %torch.cuda.device_count())
if (gpu != None):
    if torch.cuda.is_available():
        torch.cuda.set_device(CUDA_VISIBLE_DEVICES)
        print("CUDA Device: %d" %torch.cuda.current_device())

seed = 0
#--------------------------- Agent setting ------------------------------------------------------------
if dueling_dqn:
    print("use dueling dqn")
    agent = DRQNAgent(
                env=env,
                q_func=Dueling_DQN,
                optimizer_spec=optimizer,
                num_actions=num_actions,
                exploration=EXPLORATION_SCHEDULE,
                stopping_criterion=stopping_criterion,
                replay_buffer_size=REPLAY_BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                hidden_dim=HIDDEN_DIM,
                gamma=GAMMA,
                learning_starts=LEARNING_STARTS,
                learning_freq=LEARNING_FREQ,
                frame_history_len=FRAME_HISTORY_LEN,
                img_h=RESIZE_HEIGHT,
                img_w=RESIZE_WIDTH,
                img_c=3,
                target_update_freq=TARGET_UPDATE_FREQ,
                double_dqn=double_dqn,
                dueling_dqn=dueling_dqn
    )
elif dqn:
    print("use dqn")
    agent = DQNAgent(
                env=env,
                q_func=DQN,
                optimizer_spec=optimizer,
                num_actions=num_actions,
                exploration=EXPLORATION_SCHEDULE,
                stopping_criterion=stopping_criterion,
                replay_buffer_size=REPLAY_BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                learning_starts=LEARNING_STARTS,
                learning_freq=LEARNING_FREQ,
                frame_history_len=4,
                img_h=RESIZE_HEIGHT,
                img_w=RESIZE_WIDTH,
                img_c=3,
                target_update_freq=TARGET_UPDATE_FREQ,
                double_dqn=double_dqn,
                dueling_dqn=dueling_dqn
    )
elif random_play:
    print("use random player")
    agent = Ramdom(
                env=env,
                q_func=Qnetwork,
                optimizer_spec=optimizer,
                num_actions=num_actions,
                exploration=EXPLORATION_SCHEDULE,
                stopping_criterion=stopping_criterion,
                replay_buffer_size=REPLAY_BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                hidden_dim=HIDDEN_DIM,
                gamma=GAMMA,
                learning_starts=LEARNING_STARTS,
                learning_freq=LEARNING_FREQ,
                frame_history_len=FRAME_HISTORY_LEN,
                img_h=RESIZE_HEIGHT,
                img_w=RESIZE_WIDTH,
                img_c=3,
                target_update_freq=TARGET_UPDATE_FREQ,
                double_dqn=double_dqn,
                dueling_dqn=dueling_dqn
    )
elif test_model:
    print("begin to test the model")
    agent = TESTAgent(
                env=env,
                q_func=Qnetwork,
                optimizer_spec=optimizer,
                num_actions=num_actions,
                modelFile = modelFile,
                exploration=EXPLORATION_SCHEDULE,
                stopping_criterion=stopping_criterion,
                replay_buffer_size=REPLAY_BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                hidden_dim=HIDDEN_DIM,
                gamma=GAMMA,
                learning_starts=LEARNING_STARTS,
                learning_freq=LEARNING_FREQ,
                frame_history_len=FRAME_HISTORY_LEN,
                img_h=RESIZE_HEIGHT,
                img_w=RESIZE_WIDTH,
                img_c=3,
                target_update_freq=TARGET_UPDATE_FREQ,
                double_dqn=double_dqn,
                dueling_dqn=dueling_dqn
    )
else:
    print("use drqn")
    agent = DRQNAgent(
                env=env,
                q_func=Qnetwork,
                optimizer_spec=optimizer,
                num_actions=num_actions,
                exploration=EXPLORATION_SCHEDULE,
                stopping_criterion=stopping_criterion,
                replay_buffer_size=REPLAY_BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                hidden_dim=HIDDEN_DIM,
                gamma=GAMMA,
                learning_starts=LEARNING_STARTS,
                learning_freq=LEARNING_FREQ,
                frame_history_len=FRAME_HISTORY_LEN,
                img_h=RESIZE_HEIGHT,
                img_w=RESIZE_WIDTH,
                img_c=3,
                target_update_freq=TARGET_UPDATE_FREQ,
                double_dqn=double_dqn,
                dueling_dqn=dueling_dqn
    )
#--------------------------- Begin Minecraft game -----------------------------------------------------
agent.run(agent_host)
#--------------------------- Begin Minecraft game -----------------------------------------------------
print("-----------------------Training ends-----------------------")
