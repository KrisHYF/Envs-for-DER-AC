"""
Formation: N agents are tasked to position themselves equally spread out in a line between the two landmarks
Source: https://github.com/sumitsk/marl_transfer/
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from learning.envs.multiagent.core_vec import World, Agent, Landmark
from learning.envs.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2
        num_agents = 15
        num_landmarks = 2
        world.collaborative = True
        self.total_length = 4.0  # length of the line
        self.world_radius = 1.0
        self.num_others = 10
        self.np_rnd = np.random.RandomState(0)

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.03

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.02

        # make initial conditions
        interval_length = self.total_length/(num_agents - 1)
        center = (num_agents - 1)/2
        self.initial_config = np.transpose([np.array([(i - center)*interval_length, 0]) for i in range(num_agents)])
        self.episodic_config = []
        self.collide_th = 2*world.agents[0].size
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-self.world_radius, self.world_radius, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        origin = np.random.uniform(-0.25*self.world_radius, 0.25*self.world_radius, world.dim_p)
        theta = np.random.uniform(0, 2*np.pi)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        self.episodic_config = np.transpose(np.dot(rotation_matrix, self.initial_config)) + origin

        world.landmarks[0].state.p_pos = self.episodic_config[0]
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

        world.landmarks[1].state.p_pos = self.episodic_config[-1]
        world.landmarks[1].state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        rew = 0
        if agent == world.agents[0]:
            expected_poses = np.array([self.episodic_config]).repeat(len(world.agents), axis=0)
            agent_poses1 = np.array([[l.state.p_pos for l in world.agents]]).repeat(len(world.agents), axis=0)
            agent_poses2 = np.transpose(agent_poses1, axes=(1, 0, 2))
            dists = np.sqrt(np.sum(np.square(agent_poses2 - expected_poses), axis=2))
            row_ind, col_ind = linear_sum_assignment(dists)
            rew -= dists[row_ind, col_ind].sum()

            if agent.collide:
                dist_a = np.sqrt(np.sum(np.square(agent_poses1 - agent_poses2), axis=2))
                n_collide = (dist_a < self.collide_th).sum() - len(world.agents)
                rew -= n_collide

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)

        # choose closest entities
        entity_dist = np.sqrt(np.sum(np.square(np.array(entity_pos)), axis=1))
        entity_dist_idx = np.argsort(entity_dist)
        entity_pos = [entity_pos[i] for i in entity_dist_idx]

        # choose closest other agents
        other_dist = np.sqrt(np.sum(np.square(np.array(other_pos)), axis=1))
        dist_idx = np.argsort(other_dist)
        other_pos = [other_pos[i] for i in dist_idx[:self.num_others]]

        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)

        return obs

    def seed(self, seed=None):
        self.np_rnd.seed(seed)
