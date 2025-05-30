from radgraph import RadGraph, F1RadGraph

refs = [
    "no acute cardiopulmonary abnormality",
    "et tube terminates 2 cm above the carina retraction by several centimeters is recommended for more optimal placement bibasilar consolidations better assessed on concurrent chest ct",
]

hyps = [
    "no acute cardiopulmonary abnormality",
    "endotracheal tube terminates 2 5 cm above the carina bibasilar opacities likely represent atelectasis or aspiration",
]
f1radgraph = F1RadGraph(reward_level="all", cuda=0)
mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists  = f1radgraph(hyps=hyps, refs=refs)
rewards = list(zip(*reward_list))
reward1, reward2 = rewards
x = 1

