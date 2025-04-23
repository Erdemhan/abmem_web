from .ddpg import RDPG, ReplayBuffer
from .algorithm_utils import State
import numpy as np
import torch
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HYPERPARAMS
EPISODES = 3
EPOCHS = 200
TRAINS = 100
BUFFER_SIZE = 1000000
MINI_BATCH_SIZE = 64
LEARNING_RATE_ACTOR = 0.0005
LEARNING_RATE_CRITIC = 0.001
TAU = 0.0001
DISCOUNT = 0.98
NOISE_STD = 0.6
NOISE_DECAY = 0.99
NOISE_MIN = 0.01
PENALTY_DECAY = 1
PENALTY = 1500
ITERATION = 100


STATE_DIM = 4
SEQ_LEN = 24

class AgentAlgorithm:

    def __init__(self, action_dim, agent_id):
        self.agent_id = agent_id
        self.replayBuffer = ReplayBuffer(10000)
        self.ddpg = RDPG(self.replayBuffer,
                         learning_rate_actor=LEARNING_RATE_ACTOR,
                         learning_rate_critic=LEARNING_RATE_CRITIC,
                         noise=NOISE_STD, noise_decay=NOISE_DECAY,
                         noise_min=NOISE_MIN, action_dim=action_dim)
        self.failStack = 0

        # LSTM gizli durumu
        self.hidden = (
            torch.zeros(1, 1, 32).to(device),
            torch.zeros(1, 1, 32).to(device)
        )

        # Zaman dizisi için deque
        self.state_history = deque(maxlen=SEQ_LEN)
        self.next_state_history = deque(maxlen=SEQ_LEN)
        self.seq_len = SEQ_LEN




    def selectAction(self, state: State) -> list:
        current = [int(state.demand), int(state.mcp), int(state.mcp24), int(state.mcp168)]
        self.state_history.append(current)

        if len(self.state_history) < SEQ_LEN:
            random_action = np.random.uniform(0, 200, self.ddpg.action_dim)
            return random_action.tolist()

        state_seq = np.array(self.state_history)
        action, self.hidden = self.ddpg.select_action(state_seq, self.hidden)
        return action.tolist()



    def learn(self, state, action, next_state, reward, done=False):
        s = [int(state.demand), int(state.mcp), int(state.mcp24), int(state.mcp168)]
        ns = [int(next_state.demand), int(next_state.mcp), int(next_state.mcp24), int(next_state.mcp168)]
        
        self.state_history.append(s)
        self.next_state_history.append(ns)

        if len(self.state_history) < SEQ_LEN or len(self.next_state_history) < SEQ_LEN:
            return

        state_seq = np.array(self.state_history)
        next_state_seq = np.array(self.next_state_history)
        action = [int(a) for a in action]
        reward = int(reward)

        self.ddpg.add_to_replay_buffer(state_seq, action, next_state_seq, reward, done)
        self.ddpg.train(iterations=ITERATION, batch_size=MINI_BATCH_SIZE, tau=TAU, discount=DISCOUNT)
        self.ddpg.actor.update_noise()

        if done:
            self.hidden = (
                torch.zeros(1, 1, 32).to(device),
                torch.zeros(1, 1, 32).to(device)
            )
            self.state_history.clear()
            self.next_state_history.clear()

