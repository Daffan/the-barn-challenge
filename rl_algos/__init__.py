from rl_algos.td3 import TD3
from rl_algos.sac import SAC
from rl_algos.ddpg import DDPG
from rl_algos.safe_td3 import SafeTD3
from rl_algos.model_based import DynaRLAlgo, SMCPRLAlgo, MBPORLAlgo


class DynaTD3(DynaRLAlgo, TD3):
    pass

class SMCPTD3(SMCPRLAlgo, TD3):
    pass

class MBPOTD3(MBPORLAlgo, TD3):
    pass

class DynaSAC(DynaRLAlgo, SAC):
    pass

class SMCPSAC(SMCPRLAlgo, SAC):
    pass

class MBPOSAC(MBPORLAlgo, SAC):
    pass

class DynaDDPG(DynaRLAlgo, DDPG):
    pass

class SMCPDDPG(SMCPRLAlgo, DDPG):
    pass

class MBPODDPG(MBPORLAlgo, DDPG):
    pass

algo_class = {
    "TD3": TD3,
    "SAC": SAC,
    "DDPG": DDPG,
    "DynaTD3": DynaTD3,
    "DynaSAC": DynaSAC,
    "DynaDDPG": DynaDDPG,
    "SMCPTD3": SMCPTD3,
    "SMCPSAC": SMCPSAC,
    "SMCPDDPG": SMCPDDPG,
    "MBPOTD3": MBPOTD3,
    "MBPOSAC": MBPOSAC,
    "MBPODDPG": MBPODDPG,
    "SafeTD3": SafeTD3
}