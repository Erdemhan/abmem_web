class State:
    def __init__(self,mcp:int, demand: int, mcp24: int = 0, mcp168: int = 0):
        self.mcp = mcp
        self.mcp24 = mcp24
        self.mcp168 = mcp168
        self.demand = demand