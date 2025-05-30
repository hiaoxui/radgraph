import ray
from radgraph import F1RadGraph
import math


@ray.remote(num_gpus=1)
class RadGraphWorker:
    def __init__(self, reward_level="all", batch_size=8):
        self.model = F1RadGraph(reward_level=reward_level, cuda=0, batch_size=batch_size)

    def compute(self, hyps, refs):
        reward_list = self.model(hyps=hyps, refs=refs)[1]
        rewards = list(zip(*reward_list))
        return list(rewards)


class RadGraphRewardServer:
    def __init__(self, num_workers=2, reward_level="all", batch_size=8):
        ray.init(ignore_reinit_error=True)
        self.workers = [
            RadGraphWorker.remote(reward_level=reward_level, batch_size=batch_size)
            for i in range(num_workers)
        ]
        self.num_workers = num_workers

    def compute_rewards(self, hyps, refs):
        assert len(hyps) == len(refs), "hypotheses and references must be same length"
        total = len(hyps)
        batch_size = math.ceil(total / self.num_workers)

        futures = []
        for i in range(self.num_workers):
            start = i * batch_size
            end = min((i + 1) * batch_size, total)
            if start < end:
                futures.append(self.workers[i].compute.remote(hyps[start:end], refs[start:end]))

        results = ray.get(futures)
        all_rewards = [reward for batch in results for reward in batch]
        return all_rewards
