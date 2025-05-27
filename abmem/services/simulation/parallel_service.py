# parallel_service.py

import os
import random
import numpy as np
import torch
import multiprocessing

# ✅ Django ayarları en başta yüklenmeli
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_project.settings")  # <--- projenin settings dosyasını buraya yaz
import django
django.setup()


# ✅ Worker başlatıldığında seed'leri ayarla
def init_worker():
    seed = 38  # sabit seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ✅ Django importları artık güvenli
from ...models import Agent
from ...services.agent import agent_service as AgentService
from ...constants import *


def startPool(agents, algorithms):
    # 🔁 Artık agent_name ile eşleştiriyoruz
    algorithm_map = {alg.agent_name: alg for alg in algorithms}

    # 💡 agent.name üzerinden algoritmayı bul
    tasks = [(agent, algorithm_map.get(agent.name)) for agent in agents]

    with multiprocessing.Pool(initializer=init_worker) as pool:
        results = pool.starmap(AgentService.run, tasks)

    updated_algorithms, offers = zip(*results)
    return list(updated_algorithms), list(offers)

