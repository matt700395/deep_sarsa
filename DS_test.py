import copy
import pylab
import random
import numpy as np
from environment import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

class DeepSARSA(Sequential):
    def __init__(self, state_size, action_size, learning_rate):
        super().__init__()
        self.add(Dense(30, input_dim=state_size, activation='relu'))
        self.add(Dense(30, activation='relu'))
        self.add(Dense(action_size, activation='linear'))
        self.summary()
        self.compile(loss='mse', optimizer=Adam(lr=learning_rate))

# ê·¸ë¦¬?œì›”???ˆì œ?ì„œ???¥ì‚´???ì´?„íŠ¸
class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        # ?íƒœ???¬ê¸°?€ ?‰ë™???¬ê¸° ?•ì˜
        self.state_size = state_size
        self.action_size = action_size

        # ?¥ì‚´???˜ì´???Œë¼ë©”í„°
        self.lr=0.001
        self.epsilon = 0.01
        self.model = DeepSARSA(self.state_size, self.action_size, self.lr)
        self.model.load_weights('save_model/model')

    # ?…ì‹¤ë¡??ìš• ?•ì±…?¼ë¡œ ?‰ë™ ? íƒ
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # ë¬´ì‘?„í–‰??ë°˜í™˜
            return random.randrange(self.action_size)
        else:
            # ëª¨ë¸ë¡œë????‰ë™ ?°ì¶œ
            # state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

if __name__ == "__main__":
    # ?˜ê²½ê³??ì´?„íŠ¸ ?ì„±
    env = Env(render_speed=0.1)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size)

    scores, episodes = [], []

    for episode in range(10):
        done = False
        score = 0
        # env ì´ˆê¸°??
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        time = 0

        while not done:
            # ?„ì¬ ?íƒœ???€???‰ë™ ? íƒ
            action = agent.get_action(state)

            # ? íƒ???‰ë™?¼ë¡œ ?˜ê²½?ì„œ ???€?„ìŠ¤??ì§„í–‰ ???˜í”Œ ?˜ì§‘
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward
            state = next_state
            time += 1
            if time > 200:
                done = True

            if done:
                # ?í”¼?Œë“œë§ˆë‹¤ ?™ìŠµ ê²°ê³¼ ì¶œë ¥
                print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(
                      episode, score, agent.epsilon))

