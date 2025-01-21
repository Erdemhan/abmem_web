from ...models.enums import AgentState
from ...models import Agent, Offer, Portfolio, Resource
from ...services.file_reader import reader_service as ReaderService
from ...services.agent import portfolio_factory as PortfolioFactory
from ...services.agent import offer_factory as OfferFactory
from ...services.agent import agent_service as AgentService
from .ddpg import DDPG,ReplayBuffer
from .algorithm_utils import State
import numpy as np

# BASE AGENT ALGORITHM

# HYPERPARAMS
EPISODES = 3 # 0-...
EPOCHS = 200 # 0-...
TRAINS = 100 # 0-...
BUFFER_SIZE = 1000000 # 0-...
MINI_BATCH_SIZE = 64 # 0-...
LEARNING_RATE_ACTOR = 0.0005 # 0-1
LEARNING_RATE_CRITIC = 0.001 # 0-1
TAU = 0.0001 # 0-1
DISCOUNT = 0.96 # 0-1
NOISE_STD = 0.23# 0-1
NOISE_DECAY = 0.99# 0-1
NOISE_MIN = 0.001 # 0-1
PENALTY_DECAY = 1 # 1-...
PENALTY = 1500# 0-...

ITERATION = 100

class AgentAlgorithm:

    def __init__(self,action_dim):
        self.replayBuffer =  ReplayBuffer(10000)
        self.ddpg = DDPG(self.replayBuffer,learning_rate_actor=LEARNING_RATE_ACTOR,learning_rate_critic=LEARNING_RATE_CRITIC,noise=NOISE_STD,noise_decay=NOISE_DECAY,noise_min=NOISE_MIN,action_dim=action_dim)
        
    def selectAction(self, state: State) -> list:
        state = [int(state.demand),int(state.mcp)]
        action = self.ddpg.select_action(np.array(state))
        return action
    
    def learn(self, state, action, next_state, reward, done=False):
        # Add the step to the buffer
        state = [int(state.demand),int(state.mcp)]
        next_state = [int(next_state.demand),int(next_state.mcp)]
        action = [int(act) for act in action]
        reward = int(reward)
        self.ddpg.add_to_replay_buffer(state, action, next_state, reward, done)
        # Train the model
        self.ddpg.train(iterations= ITERATION, batch_size=MINI_BATCH_SIZE, tau=TAU, discount=DISCOUNT)
        self.ddpg.actor.update_noise()
