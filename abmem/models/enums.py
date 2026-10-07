from django_enumfield import enum

class EnergyType(enum.Enum):
    RENEWABLE = 0
    FOSSIL = 1
    NUCLEAR = 2

class PeriodType(enum.Enum):
    HOUR = 0
    DAY = 1
    WEEK = 2
    MONTH = 3
    YEAR = 4

class SimulationState(enum.Enum):
    CREATED = 0
    STARTED = 1
    INITIALIZED = 2
    BIDDING = 3
    MARKETCLEARING = 4
    PERIODRESULT= 5
    RESULT= 6
    FINISHED = 7
    STOPPED = 8
    CANCELED = 9
    FAILED = 10

class SimulationMode(enum.Enum):
    PERIODBYPERIOD = 0
    ONLYRESULT = 1


class MarketStrategy(enum.Enum):
    PAYASBID = 0
    PAYASPTF = 1


class MarketState(enum.Enum):
    CREATED = 0
    INITIALIZING = 1
    INITIALIZED = 2
    WAITINGAGENTS = 3
    CALCULATING = 4
    MARKETCLEARING = 5
    BROADCASTING = 6
    PERIODEND = 7

class AgentState(enum.Enum):
    CREATED = 0
    WAITING = 1
    RUNNING= 2
    LEARNING = 3
    PREDICTING = 4
    CALCULATING = 5
    OFFERING = 6
    INITIALIZED = 7

class AgentType(enum.Enum):
    RENEWABLE = 0
    FOSSIL = 1
    NUCLEAR = 2
    HYBRID = 3


class AlgorithmType(enum.Enum):
    DDPG = 0
    RDPG = 1
    TD3 = 2
    SAC = 3
    PPO = 4
    RANDOM = 5
    MARGINAL_COST = 6
    FIXED_MARKUP = 7


class HPOStatus(enum.Enum):
    CREATED = 0
    RUNNING = 1
    FINISHED = 2
    FAILED = 3
    CANCELED = 4


class HPOPhase(enum.Enum):
    INIT = 0
    EMPLOYED = 1
    ONLOOKER = 2
    SCOUT = 3
    FINAL_EVAL = 4


class HPOTrialStatus(enum.Enum):
    PENDING = 0
    RUNNING = 1
    FINISHED = 2
    FAILED = 3
