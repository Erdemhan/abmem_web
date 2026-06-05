# abm_ddpg/config.py
ITERATION = 5         # Aşırı öğrenmeyi önler, daha stabil
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-3
TAU = 0.005             # Daha dengeli güncelleme
DISCOUNT = 0.99
NOISE_STD = 0.6        # Daha az agresif keşif
NOISE_DECAY = 0.995     # Daha yavaş azalsın
NOISE_MIN = 0.05        # Keşif bitmesin ama azalsın
SEQ_LEN = 24