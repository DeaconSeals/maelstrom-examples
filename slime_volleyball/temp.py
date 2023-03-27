import slimevolleygym
import gym

env = gym.make("SlimeVolley-v0")
obs1 = env.reset()
print(obs1)
