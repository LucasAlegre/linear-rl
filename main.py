import gym
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda


def main():

    env = gym.make('Acrobot-v1')
    agent = TrueOnlineSarsaLambda(env.observation_space, env.action_space)

    obs = env.reset()
    ret = 0
    while True:
        action = agent.act(obs)
        new_obs, rew, done, info = env.step(action)
        ret += rew

        agent.learn(obs, action, rew, new_obs, done)

        obs = new_obs
        if done:
            print("Return:", ret)
            ret = 0
            obs = env.reset()
        env.render()

if __name__ == '__main__':
    main()