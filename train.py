import option_critic
import gym


if __name__=="__main__":

    env = gym.make("MountainCar-v0")
    policy = option_critic.network.PolicyNet(2, env.action_space.n,
                                             option_critic.network.FCBody(env.observation_space.shape[0]))

    critic = option_critic.network.VanillaNet(2, option_critic.network.FCBody(env.observation_space.shape[0]))

    agent = option_critic.OptionCritic(env, policy, critic)

    agent.learn(100)