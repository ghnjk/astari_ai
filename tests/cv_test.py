#!/usr/bin/python
# -*- coding: UTF-8 -*-
import gym
import matplotlib.pyplot as plt
import cv2


def test_space_invador_img():
    env = gym.make("SpaceInvaders-v0")
    s = env.reset()
    print("observation shape: ", s.shape)
    plt.imshow(s)
    plt.show()
    s = s[25: -12, :, :]
    ns = cv2.resize(cv2.cvtColor(s, cv2.COLOR_RGB2GRAY), (84, 84))
    print("new state shape: ", ns.shape)
    plt.imshow(ns, cmap="gray")
    plt.show()


if __name__ == '__main__':
    test_space_invador_img()
