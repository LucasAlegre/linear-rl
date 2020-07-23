import gym
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from linear_rl.fourier import FourierBasis
import matplotlib.pyplot as plt


def main():

    env = gym.make('Acrobot-v1')
    agent = TrueOnlineSarsaLambda(env.observation_space, env.action_space,
                                alpha=0.0001,
                                fourier_order=3)

    obs = env.reset()
    ret = 0
    rets = []
    episodes = 400
    ep =  0
    while ep < episodes:
        action = agent.act(obs)
        new_obs, rew, done, info = env.step(action)
        ret += rew

        agent.learn(obs, action, rew, new_obs, done)

        obs = new_obs
        if done:
            print("Return:", ret)
            rets.append(ret)
            ret = 0
            ep += 1
            obs = env.reset()
        #env.render('human')
    
    plt.figure()
    plt.plot(rets)
    plt.ylabel("Return")
    plt.xlabel("Episode")
    plt.show()

if __name__ == '__main__':
    main()
