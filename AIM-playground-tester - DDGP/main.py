import os
import sys

import pyglet
from env import ArmEnv
from rl_v2 import DDPG

MAX_EPISODES = 2500
MAX_EP_STEPS = 300
CONTINUE_TRAINING = False

# Disable GPU computation
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []

def on_key_press(symbol, _):
    if symbol == pyglet.window.key.ESCAPE:
        env.viewer.close()

def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            env.render()
            a = rl.choose_action(s)
            s_, r, done = env.step(a)
            rl.store_transition(s, a, r, s_)
            ep_r += r
            if rl.memory_full:
                # start to learn once memory is full
                rl.learn()
            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: % 6.1f | step: %i' % (i, '----' if not done else 'done', ep_r, j))
                break
        if i % 50 == 0 and i > 0:
            rl.save()
    rl.save()

def eval_single_example(step_limit=400):
    rl.restore()
    s = env.reset()
    step = 0
    while step < step_limit:
        step += 1
        env.render(follow_mouse=False)
        a = rl.choose_action(s)
        s, r, done = env.step(a)

        if done:
            print(f"Target hit after {step} steps")
            return 1

    print(f"Unable to hit target in within {step_limit} steps")
    return 0



def eval():
    rl.restore()
    env.render()
    window = env.viewer
    window.set_vsync(True)
    window.push_handlers(on_key_press)
    cursor = window.get_system_mouse_cursor(pyglet.window.Window.CURSOR_HAND)
    window.set_mouse_cursor(cursor)
    s = env.reset()
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done = env.step(a)


def run():
    if sys.argv[1] == 'TRAIN':
        if CONTINUE_TRAINING:
            rl.restore()
        train()
    elif sys.argv[1] == 'TEST_SINGLE_EXAMPLE':
        eval_single_example()
    else:
        eval()

if __name__ == '__main__':
    run()
