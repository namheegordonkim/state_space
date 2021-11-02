from imitation.algorithms import bc


# https://stable-baselines.readthedocs.io/en/master/guide/pretrain.html
# https://github.com/HumanCompatibleAI/imitation


def train(env,transitions,n_epochs):

    bc_trainer = bc.BC(
        observation_space = env.observation_space,
        action_space = env.action_space,
        demonstrations = transitions
    )
    bc_trainer.train(n_epochs=n_epochs)

def create_dataset():
    pass

def main():
    pass


if __name__ == "__main__":
    main()