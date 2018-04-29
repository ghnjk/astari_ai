[space-invader gym](https://gym.openai.com/envs/SpaceInvaders-v0/)

# 网络结构

```
state: 210 * 160 * 3
action: 1 [1, 6]

network:

3 * img_state
reward = reward

conv2d(16, (3,3) , kernel=3,3 stride=1,1 )
batch_normal
relu
conv2d(16, (3,3) , kernel=3,3 stride=1,1 )
batch_normal
relu

maxppol  stride(2, 2)
dropout

conv2d(32, (3,3) , kernel=3,3 stride=1,1 )
batch_normal
relu
conv2d(32, (3,3) , kernel=3,3 stride=1,1 )
batch_normal
relu

maxppol  stride(2, 2)
dropout

flatten()
dense(256)
relu
--------------------------
dense(64)
relu
dense 1       --> state_value
---------------------------
dense(64)
relu
dense 6        --> action advantage
-------------------------
q = state_value + (action_advantage - action_aadavantage_mean)

```
