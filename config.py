#!/usr/bin/python
# -*- coding: UTF-8 -*-

game_config = {
    "GAME_NAME": "Breakout-v0",
    "SAME_ACTION_STEP": 1,
    "TIME_STEP_AS_STATE": 4,
    "IMAGE_HEIGHT": 84,
    "IMAGE_WIDTH": 84,
    "IS_DUEL": True,
    "WEIGHT_DATA_PATH": "data/duel_dqn_weights/duel_dqn_weights.ckpt",
    "LOG_PATH": "logs/duel_dqn"
}

train_config = {
    "MEMORY_SIZE": 50000,
    "BATCH_SIZE": 64,
    "UPDATE_TARGET_ITER": 10000,
    "LEARNING_RATE": 0.0002,
    "CHOOSE_E_GREEDY_INCREASEMENT": 0.00002
}
