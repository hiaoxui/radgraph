from radgraph import RadGraphRewardServer


if __name__ == "__main__":
    refs = [
        "no acute cardiopulmonary abnormality",
        "et tube terminates 2 cm above the carina retraction by several centimeters is recommended for more optimal placement bibasilar consolidations better assessed on concurrent chest ct",
    ] * 10
    hyps = [
        "no acute cardiopulmonary abnormality",
        "endotracheal tube terminates 2 5 cm above the carina bibasilar opacities likely represent atelectasis or aspiration",
    ] * 10

    server = RadGraphRewardServer(num_workers=2)
    rewards = server.compute_rewards(hyps, refs)
    for i, reward in enumerate(rewards):
        print(f"Pair {i+1} Rewards: {reward}")
