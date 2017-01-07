# Breakout-Q

A Q-learning Agent which plays breakout well.

The breakout game is based on [CoderDojoSV/beginner-python](https://github.com/CoderDojoSV/beginner-python)'s tutorial, and the Q-learning implementation is inspired by [SarvagyaVaish/FlappyBirdRL](http://sarvagyavaish.github.io/FlappyBirdRL/)

## About

To start, run:

```
python game.py
```

By default, the script will load the trained Q array from file named **trainedQ_breakout.npz**.

If you want to train your own agent, please rename **trainedQ_breakout.npz**, and run the script. Then it will initialize a new Q array. After closing the game window, the data will be saved to **trainedQ_breakout.npz** automatically.

## Dependency

The following python package is necessary to run the script:

- numpy
- pygame
