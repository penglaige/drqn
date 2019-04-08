from __future__ import print_function
from __future__ import division

import cv2
import numpy as np
from collections import deque

from builtins import range
from past.utils import old_div
import MalmoPython
import random
import time
import logging
import struct
import socket
import os
import sys
import malmoutils
import json

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ENV():
    def __init__(self, agent_host, missionXML, validate, my_mission_record, logger,recordingsDirectory, MAX_EPISODE):
        self.agent_host = agent_host
        self.missionXML = missionXML
        self.validate = validate
        self.my_mission_record = my_mission_record
        self.logger = logger
        self.cumulative_rewards = 0.0
        self.steps = 0
        #self.cumulative_rewards_from_agent = 0.0
        self.episode_rewards = []
        self.episode_steps_average_rewards = []
        self.max_episode = MAX_EPISODE
        self.recordingsDirectory=recordingsDirectory
        self.MAX_EPISODE = MAX_EPISODE
        self.MAX_STEP = 100
        self.MAX_APPLE = 5
        self.width = None
        self.height = None
        self.channels = None
        self.apple_num = 0
        self.total_apple = []
        #-------random--------
        #self.apple_space = self.get_random_space(0)
        #self.agent_space = self.get_random_space(1)
        #self.apple_x,self.apple_z,self.agent_x,self.agent_z,self.agent_yaw = self.get_random_position()
        #-------fix-------
        #_,_,_,_,agent_yaw = self.get_random_position()
        #self.my_mission.drawItem( 48,5,48,"apple")
        #self.my_mission.startAtWithPitchAndYaw( 43.5,5,43.5,30,agent_yaw)

    def get_random_position(self, object):
        if object == "agent":
            # agent x,z~[42,49], y =5, yaw = [0,360)
            x = random.randint(42,49)
            y = 5
            z = random.randint(42,49)
            yaw = random.randint(0,3) * 90
            return x,y,z,yaw
        elif object == "items":
            # item x,z~(42,49) y = 5
            x = random.randint(42,49)
            y = 5
            z = random.randint(42,49)
            return x,y,z
        else:
            return "Error!"

        # --------------------------------------------------

    def canset(self):
        #print("canset",len(self.episode_rewards),self.MAX_EPISODE)
        if len(self.episode_rewards) < self.MAX_EPISODE:
            return 1
        else:
            return 0

    def reset(self):
        #  ------ run mission ---------
        if self.recordingsDirectory:
            self.my_mission_record.setDestination(self.recordingsDirectory + "//" + "Mission_" + str(len(self.episode_rewards) + 1) + ".tgz")

        # get random agent position
        agent_x,agent_y,agent_z,agent_yaw = self.get_random_position("agent")
        mission = MalmoPython.MissionSpec(self.missionXML, self.validate)

        mission.startAtWithPitchAndYaw(agent_x,agent_y,agent_z,30,agent_yaw)
        print("agent position: x:{},y:{},z:{},yaw:{}".format(agent_x,agent_y,agent_z,agent_yaw))
        for i in range(self.MAX_APPLE):
            item_x,item_y,item_z = self.get_random_position("items")
            mission.drawItem(item_x,item_y,item_z,"apple")

        mission.forceWorldReset()

        time.sleep(0.5)
        self.agent_host.sendCommand("quit")
        max_retries = 3
        for retry in range(max_retries):
            try:
                self.agent_host.startMission( mission, self.my_mission_record )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    self.logger.error("Error starting mission: %s" % e)
                    exit(1)
                else:
                    time.sleep(2)

        self.logger.info('Mission %s', len(self.episode_rewards)+1)
        self.logger.info("Waiting for the mission to start")
        world_state = self.agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = self.agent_host.getWorldState()

        # game start
        self.agent_host.sendCommand("hotbar.4 1")
        self.steps = 0
        self.cumulative_rewards = 0.0
        self.apple_num = 0
        mission = None
        frame = self.get_frame()
        return frame


    def get_frame(self,ALL=False):
        world_state = self.agent_host.getWorldState()
        while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
            #self.logger.info("Waiting for frames...")
            time.sleep(0.05)
            world_state = self.agent_host.getWorldState()


        if len(world_state.video_frames) > 0:
            #self.logger.info("Got frame!")
            if not self.width or not self.height or not self.channels:
                self.width = world_state.video_frames[0].width
                self.height = world_state.video_frames[0].height
                self.channels = world_state.video_frames[0].channels
            frame = world_state.video_frames[0].pixels #frame h w C
        elif not world_state.is_mission_running:
            self.logger.info("Episode end!")
            frame = np.zeros((self.height,self.width,self.channels))

        frame = np.array(frame).reshape(self.height,self.width,self.channels)
        if not ALL:
            return frame

        reward = self.get_reward(world_state)
        self.cumulative_rewards += reward

        done = self.done(world_state)

        if done:
            if self.steps == 0:
                return False
            print("collected apples: %f" % self.apple_num)
            print("cumulative_rewards: %f" % self.cumulative_rewards)
            self.episode_rewards.append(self.cumulative_rewards)
            print("average_rewards: %f" % (self.cumulative_rewards / self.steps))
            self.episode_steps_average_rewards.append(self.cumulative_rewards / self.steps)
            self.total_apple.append(self.apple_num)
            self.cumulative_rewards = 0.0
            self.apple_num = 0
            self.steps = 0

        return frame, reward, done

    def done(self,world_state):
        done_from_time = not world_state.is_mission_running
        done_from_task = False

        if self.apple_num >= self.MAX_APPLE or self.steps >= self.MAX_STEP:
            done_from_task = True
            if world_state.is_mission_running:
                self.agent_host.sendCommand(" move 0")
                for i in range(1):
                    print("try to quit mission...")
                    time.sleep(0.04)
        return done_from_time or done_from_task



    def get_reward(self,world_state):
        reward_from_obs = 0
        reward_from_sight = 0

        if len(world_state.observations) > 0:
            msg = world_state.observations[-1].text
            information = json.loads(msg)
            new_apple_num = information.get(u'InventorySlot_0_size', 0)
            LineOfSight = information.get(u'LineOfSight', 0)
            reward_from_obs = 10 * (new_apple_num - self.apple_num)
            #if LineOfSight:
            #    if LineOfSight["type"] == "apple" and LineOfSight["distance"] <= 1.3:
            #        print("See apple!")
            #        reward_from_sight = 50

        if len(world_state.rewards) < 1:
            reward_from_agent = 0
        else:
            reward_from_agent = world_state.rewards[-1].getValue()
        if reward_from_agent != reward_from_obs:
            reward = max(reward_from_agent, reward_from_obs)
            self.apple_num += reward / 10
            return reward
        else:
            self.apple_num += reward_from_agent / 10
            return reward_from_agent


    def step(self, action):
        self.steps += 1
        if action == 0:
            # turn left
            self.agent_host.sendCommand("turn -1")

        elif action == 1:
            # turn right
            self.agent_host.sendCommand("turn 1")

        elif action == 2:
            # go straight
            self.agent_host.sendCommand("move 1")

        new_obs, reward, done = self.get_frame(ALL=True)
        return new_obs, reward, done

    def get_episode_rewards(self):
        # total rewards
        return self.episode_rewards

    def get_average_rewards_per_episode(self):
        # average rewards
        return self.episode_steps_average_rewards
