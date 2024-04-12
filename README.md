# Tests of Custom Logger with PPO training

Enviroments tested are [Pendulum](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py) and [CartPole](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py), implementation on [Gymansium github](https://github.com/Farama-Foundation/Gymnasium)

The training with PPO of [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html) are done quickly and are not sufficient, but the aim of this code is to show the implementation and use of a custom callback for tensorboard (see [here](https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html) for more info), saving and loading a trained model (see [here](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#basic-usage-training-saving-loading) for more info).

## Requirements

To run this repo the following library are necessary: [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html), [Gymansium](https://github.com/Farama-Foundation/Gymnasium), [Tensorboard](https://www.tensorflow.org/tensorboard)

## Train Model
The model is trained using [trainCartPole.py](trainCartPole.py) and [trainPendulum.py](trainPendulum.py) 


## Test Trained Model
The test of trained model is done executing [testmodel.py](testmodel.py).
