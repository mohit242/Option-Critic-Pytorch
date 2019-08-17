import option_critic
import gym
import gym_minigrid

if __name__=="__main__":

    # env = gym.make("Taxi-v2")
    # if isinstance(env.observation_space, gym.spaces.Discrete):
    #     env = option_critic.DiscreteToBox(env)
    env = gym.make("MiniGrid-FourRooms-v0")
    env = option_critic.FlattenArrayObservation(env)
    input_dim = env.observation_space.shape[0]
    policy = option_critic.network.PolicyNet(4, env.action_space.n,
                                             option_critic.network.FCBody(input_dim))

    agent = option_critic.OptionCritic(env, policy)

    agent.learn(100000)