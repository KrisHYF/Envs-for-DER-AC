"""
Scenario: Cooperative coverage of six agents
Note: In the individual observation, the relative position lists are sorted.
"""
import numpy as np
from learning.envs.multiagent.core_vec import World, Agent, Landmark
from learning.envs.multiagent.scenario import BaseScenario
from bridson import poisson_disc_samples


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2
        num_agents = 6
        num_landmarks = 6
        world.collaborative = True
        self.world_radius = 1.5  # camera range
        self.np_rnd = np.random.RandomState(0)  # 伪随机数生成器 (not used here)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.collision_threshold = 2 * world.agents[0].size
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
            agent.state.p_pos = np.random.uniform(-self.world_radius, +self.world_radius, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # generate locations of the landmarks
        l_locations = poisson_disc_samples(width=self.world_radius*2, height=self.world_radius*2, r=0.15*4.5)
        while len(l_locations) < len(world.landmarks):
            l_locations = poisson_disc_samples(width=self.world_radius*2, height=self.world_radius*2, r=0.15*4.5)
            # print('regenerate l location')
        l_locations = np.array(l_locations)

        idx_list = np.random.choice(len(l_locations), len(world.landmarks), replace=False)
        for i, landmark in enumerate(world.landmarks):
            idx_i = idx_list[i]
            landmark.state.p_pos = l_locations[idx_i, :] - self.world_radius
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        """
        Vectorized reward function
        Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        """
        rew = 0

        if agent == world.agents[0]:
            l_pos = np.array([[l.state.p_pos for l in world.landmarks]]).repeat(len(world.agents), axis=0)
            a_pos = np.array([[a.state.p_pos for a in world.agents]])
            a_pos1 = a_pos.repeat(len(world.agents), axis=0)
            a_pos1 = np.transpose(a_pos1, axes=(1, 0, 2))
            a_pos2 = a_pos.repeat(len(world.agents), axis=0)
            dist = np.sqrt(np.sum(np.square(l_pos - a_pos1), axis=2))
            rew = np.min(dist, axis=0)
            rew = -np.sum(rew)
            if agent.collide:
                dist_a = np.sqrt(np.sum(np.square(a_pos1 - a_pos2), axis=2))
                n_collide = (dist_a < self.collision_threshold).sum() - len(world.agents)
                rew -= n_collide

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # get positions of other agents in this agent's reference frame
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
        other_pos = [other_pos[i] for i in dist_idx]

        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)

        return obs

    def seed(self, seed=None):
        self.np_rnd.seed(seed)
