from .RDPG import RDPG, ReplayBuffer as RDPGReplayBuffer
from .DDPG import DDPG, ReplayBuffer as DDPGReplayBuffer
from .TD3 import TD3, ReplayBuffer as TD3ReplayBuffer
from .ppo import PPO, RolloutBuffer as PPORolloutBuffer
from .SAC import SAC,ReplayBuffer as SACReplayBuffer
from .algorithm_utils import State
import numpy as np
import torch
from collections import deque


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANT PARAMS
STATE_DIM = 4
MINI_BATCH_SIZE = 32
BUFFER_SIZE = 1000    # Büyük ama aşırı değil
MINI_BATCH_SIZE = 64    # Gayet uygun
NOISE_MIN = 0.01

# COMMON HYPERPARAMETERS
LEARNING_RATE_ACTOR = 0.0005
LEARNING_RATE_CRITIC = 0.001
TAU = 0.01
DISCOUNT = 0.98
NOISE_STD = 0.6
NOISE_DECAY = 0.99
ITERATION = 10

# RDPG
SEQ_LEN = 24

#PPO
PPO_GAMMA = 0.99 # GAE için
PPO_GAE_LAMBDA = 0.95 # GAE lambda
PPO_CLIP_PARAM = 0.2 # PPO clipping
PPO_PPO_EPOCHS = 10

#SAC
LEARNING_RATE_ALPHA = 0.0001  # SAC alpha learning rate 
INITIAL_ALPHA = 0.2  # SAC entropy coefficient 


class AgentAlgorithm:

    def __init__(self, action_dim, agent_name,params=None,):  
        self.agent_name = agent_name
        self.algorithm_type = params.get("ALGORITHM", "RDPG")
        if self.algorithm_type not in ['RDPG', 'DDPG', 'TD3', 'PPO', 'SAC']:
            raise ValueError(f"Unsupported algorithm type: {self.algorithm_type}")

        global ITERATION, DISCOUNT, TAU, LEARNING_RATE_ACTOR, LEARNING_RATE_CRITIC, NOISE_STD, NOISE_DECAY
        global SEQ_LEN, PPO_GAMMA, PPO_GAE_LAMBDA, PPO_CLIP_PARAM, PPO_PPO_EPOCHS,LEARNING_RATE_ALPHA,INITIAL_ALPHA

        ITERATION = params["ITERATION"]
        DISCOUNT = params["DISCOUNT"]
        TAU = params["TAU"]
        LEARNING_RATE_ACTOR = params["LEARNING_RATE_ACTOR"]
        LEARNING_RATE_CRITIC = params["LEARNING_RATE_CRITIC"]
        NOISE_STD = params["NOISE_STD"]
        NOISE_DECAY = params["NOISE_DECAY"]
        self.failStack = 0

        if self.algorithm_type == 'RDPG':
            self.replayBuffer = RDPGReplayBuffer(BUFFER_SIZE)
            self.algorithm = RDPG(self.replayBuffer,
                            learning_rate_actor=LEARNING_RATE_ACTOR,
                            learning_rate_critic=LEARNING_RATE_CRITIC,
                            noise=NOISE_STD, noise_decay=NOISE_DECAY,
                            noise_min=NOISE_MIN, action_dim=action_dim)
            self.hidden = (
                torch.zeros(1, 1, 32).to(device),
                torch.zeros(1, 1, 32).to(device)
            )
            self.state_history = deque(maxlen=SEQ_LEN)
            self.next_state_history = deque(maxlen=SEQ_LEN)
            self.seq_len = SEQ_LEN

        elif self.algorithm_type == 'DDPG':
            self.replayBuffer = DDPGReplayBuffer(BUFFER_SIZE)
            self.algorithm = DDPG(self.replayBuffer,learning_rate_actor=LEARNING_RATE_ACTOR,learning_rate_critic=LEARNING_RATE_CRITIC,noise=NOISE_STD,noise_decay=NOISE_DECAY,noise_min=NOISE_MIN,action_dim=action_dim)

        elif self.algorithm_type == 'TD3':
            self.replayBuffer = TD3ReplayBuffer(BUFFER_SIZE)
            self.algorithm = TD3(
                self.replayBuffer,
                learning_rate_actor=LEARNING_RATE_ACTOR,
                learning_rate_critic=LEARNING_RATE_CRITIC,
                noise=NOISE_STD,
                noise_decay=NOISE_DECAY,
                noise_min=NOISE_MIN,
                action_dim=action_dim
            )
        
        elif self.algorithm_type == 'PPO':
            PPO_GAMMA = params.get("PPO_GAMMA", 0.99) # GAE için
            PPO_GAE_LAMBDA = params.get("PPO_GAE_LAMBDA", 0.95) # GAE lambda
            PPO_CLIP_PARAM = params.get("PPO_CLIP_PARAM", 0.2) # PPO clipping
            PPO_PPO_EPOCHS = params.get("PPO_PPO_EPOCHS", 10) # PPO epochs




            self.replayBuffer = PPORolloutBuffer(BUFFER_SIZE)
            self.algorithm = PPO(
                self.replayBuffer,
                action_dim=action_dim,
                learning_rate_actor=LEARNING_RATE_ACTOR,
                learning_rate_critic=LEARNING_RATE_CRITIC,
                gamma=PPO_GAMMA,
                gae_lambda=PPO_GAE_LAMBDA,
                clip_param=PPO_CLIP_PARAM,
                ppo_epochs=PPO_PPO_EPOCHS,
                noise=NOISE_STD,
                noise_decay=NOISE_DECAY,
                noise_min=NOISE_MIN
            )

        elif self.algorithm_type == 'SAC':
            LEARNING_RATE_ALPHA = params.get("LEARNING_RATE_ALPHA")
            INITIAL_ALPHA = params.get("INITIAL_ALPHA")

            self.replayBuffer = SACReplayBuffer(BUFFER_SIZE)
            self.algorithm = SAC(
                self.replayBuffer,
                    action_dim=action_dim,
                    learning_rate_actor=LEARNING_RATE_ACTOR,
                    learning_rate_critic=LEARNING_RATE_CRITIC,
                    learning_rate_alpha=LEARNING_RATE_ALPHA,
                    initial_alpha=INITIAL_ALPHA,
                    tau=TAU,
                    gamma=DISCOUNT
            )

    def selectAction(self, state: State) -> list:
        if self.algorithm_type == 'RDPG':
            current = [int(state.demand), int(state.mcp), int(state.mcp24), int(state.mcp168)]
            self.state_history.append(current)
            if len(self.state_history) < SEQ_LEN:
                random_action = np.random.randint(0, 200, self.algorithm.action_dim)
                return random_action.tolist()

            state_seq = np.array(self.state_history)
            action, self.hidden = self.algorithm.select_action(state_seq, self.hidden)
            return action.tolist()
        
        else:
            state = [int(state.demand),int(state.mcp),int(state.mcp24),int(state.mcp168)]
            action = self.algorithm.select_action(np.array(state))
            if self.algorithm_type == 'TD3' or self.algorithm_type == 'PPO':
                return action
            return action.tolist()



    def learn(self, state, action, next_state, reward, done=False):
        if self.algorithm_type == 'RDPG':
            state_array = [int(state.demand), int(state.mcp), int(state.mcp24), int(state.mcp168)]
            next_state_array = [int(next_state.demand), int(next_state.mcp), int(next_state.mcp24), int(next_state.mcp168)]

            self.state_history.append(state_array)
            self.next_state_history.append(next_state_array)

            if len(self.state_history) < SEQ_LEN or len(self.next_state_history) < SEQ_LEN:
                return

            state_seq = np.array(self.state_history)
            next_state_seq = np.array(self.next_state_history)
            action = [int(a) for a in action]
            reward = int(reward)

            self.algorithm.add_to_replay_buffer(state_seq, action, next_state_seq, reward, done)
            self.algorithm.train(iterations=ITERATION, batch_size=MINI_BATCH_SIZE, tau=TAU, discount=DISCOUNT)
            self.algorithm.actor.update_noise()

        else:
            # Add the step to the buffer
            state = [int(state.demand),int(state.mcp),int(state.mcp24),int(state.mcp168)]
            next_state = [int(next_state.demand),int(next_state.mcp),int(next_state.mcp24),int(next_state.mcp168)]
            action = [int(act) for act in action]
            reward = int(reward)

            if self.algorithm_type == 'DDPG':
                self.algorithm.add_to_replay_buffer(state, action, next_state, reward, done)
                # Train the model
                self.algorithm.train(iterations= ITERATION, batch_size=MINI_BATCH_SIZE, tau=TAU, discount=DISCOUNT)
                self.algorithm.actor.update_noise()

            elif self.algorithm_type == 'TD3':
                self.algorithm.add_to_replay_buffer(state, action, next_state, reward, done)
                # TD3 eğitimi
                self.algorithm.train(iterations= ITERATION, batch_size=MINI_BATCH_SIZE, tau=TAU, discount=DISCOUNT)
                # TD3 için failstack güncelleme
                success = reward > 0  # Başarı kriterini reward'a göre belirle
                self.algorithm.update_failstack(success)
                # TD3 için noise güncelleme
                self.algorithm.actor.update_noise()
                # Üst seviye failStack'i senkronize et
                self.failStack = self.algorithm.failstack

            elif self.algorithm_type == 'PPO':
                # PPO eğitimi
                self.algorithm.add_to_rollout_buffer(state, action, next_state, reward, done)
                self.algorithm.train(iterations=ITERATION, batch_size=MINI_BATCH_SIZE, discount=DISCOUNT)
                # PPO için failstack güncelleme
                success = reward > 0
                self.algorithm.update_failstack(success)
                # PPO için risk tolerance güncelleme
                self.algorithm.policy.update_risk_tolerance(reward, action[0])
                # Üst seviye failStack'i senkronize et
                self.failStack = self.algorithm.failstack
                
            elif self.algorithm_type == 'SAC':
                self.algorithm.add_to_replay_buffer(state, action, next_state, reward, done)
                buffer_size = len(self.replayBuffer.storage)
                if buffer_size >= MINI_BATCH_SIZE:
                    self.algorithm.train(iterations=ITERATION, batch_size=MINI_BATCH_SIZE)
                else:
                    pass

