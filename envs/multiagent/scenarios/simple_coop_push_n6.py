"""
Cooperative Push: N cooperating agents are tasked to push a large ball to a target position.
Note: Smaller world size, landmark mass, larger agent size, landmark size, reward parameters.
"""
import numpy as np
from learning.envs.multiagent.core_vec import World, Agent, Landmark
from learning.envs.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.dim_c = 2
        num_agents = 6
        num_landmarks = 2
        world.collaborative = True
        self.world_radius = 0.7  # default: 1.0
        self.np_rnd = np.random.RandomState(0)
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.06  # default: 0.04
        # add landmarks (one large ball and a target location)
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            if i < num_landmarks / 2:
                landmark.name = 'landmark %d' % i
                landmark.collide = True
                landmark.movable = True
                landmark.size = 0.15  # default: 0.1
                landmark.initial_mass = 3.0  # default: 4.0
            else:
                landmark.name = 'target %d' % (i - num_landmarks / 2)
                landmark.collide = False
                landmark.movable = False
                landmark.size = 0.05
                landmark.initial_mass = 3.0  # default: 4.0
        # make initial conditions
        self.color = {'green': np.array([0.35, 0.85, 0.35]), 'blue': np.array([0.35, 0.35, 0.85]), 'red': np.array([0.85, 0.35, 0.35]),
                      'light_blue': np.array([0.35, 0.85, 0.85]), 'yellow': np.array([0.85, 0.85, 0.35]), 'black': np.array([0.0, 0.0, 0.0])}
        self.collide_th = world.agents[0].size + world.landmarks[0].size
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.0, 0.0, 0.0])

        # random properties for landmarks
        color_keys = list(self.color.keys())
        for i, landmark in enumerate(world.landmarks):
            if i < len(world.landmarks) / 2:
                landmark.color = self.color[color_keys[i]] - 0.1
            else:
                landmark.color = self.color[color_keys[int(i / 2)]] + 0.1

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-(self.world_radius - 0.2), self.world_radius - 0.2, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        num_landmark = int(len(world.landmarks) / 2)
        for i, landmark in enumerate(world.landmarks[:num_landmark]):
            landmark.state.p_pos = np.random.uniform(-(self.world_radius - 0.2), self.world_radius - 0.2, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        for i, target in enumerate(world.landmarks[num_landmark:]):
            target.state.p_pos = np.random.uniform(-(self.world_radius - 0.2), self.world_radius - 0.2, world.dim_p)
            while np.sqrt(np.sum(np.square(target.state.p_pos - world.landmarks[i].state.p_pos))) < 0.7: # default: 0.8
                target.state.p_pos = np.random.uniform(-(self.world_radius - 0.2), self.world_radius - 0.2, world.dim_p)
            target.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        if agent == world.agents[0]:
            l, t = world.landmarks[0], world.landmarks[1]
            a_pos = np.array([[a.state.p_pos for a in world.agents]])
            dist = np.sqrt(np.sum(np.square(l.state.p_pos - t.state.p_pos)))
            rew -= 4 * dist  # default: 2, large ball should move to the target
            dist2 = np.sqrt(np.sum(np.square(a_pos - l.state.p_pos), axis=2))
            rew -= 0.2 * np.mean(dist2)  # default: 0.1 and min, small balls should be close to the large ball
            n_collide = (dist2 < self.collide_th).sum()
            rew += 0.2 * n_collide  # default: 0.1, small balls should increase collisions with large balls
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

        # choose closest other agents
        other_dist = np.sqrt(np.sum(np.square(np.array(other_pos)), axis=1))
        dist_idx = np.argsort(other_dist)
        other_pos = [other_pos[i] for i in dist_idx]

        obs = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)

        return obs

    def seed(self, seed=None):
        self.np_rnd.seed(seed)
