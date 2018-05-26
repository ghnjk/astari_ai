[Breakout gym](https://gym.openai.com/envs/Breakout-ram-v0/)

  http://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf
  https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

# 网络结构

```
image network

image(84, 84, 4)
conv_2d(32, 8, 4)
relu
conv_2d(64, 4, 2)
relu
conv_2d(64, 3, 1)
relu
flatten()
dense(512)
dense(action_count)

loss = mse(q_target - q_eval)
```
