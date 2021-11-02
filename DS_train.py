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
        self.discount_factor = 0.99
        self.lr = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .999
        self.epsilon_min = 0.01
        self.model = DeepSARSA(self.state_size, self.action_size, self.lr)

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

    # <s, a, r, s', a'>???˜í”Œë¡œë???ëª¨ë¸ ?…ë°?´íŠ¸
    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # ANN?™ìŠµ
        target_Q = self.model.predict(state)[0]
        next_Q = self.model.predict(next_state)[0][next_action]

        # ?´ì‚¬???í•¨???…ë°?´íŠ¸ ??
        if done:
            target_Q[action] = reward
        else:
            target_Q[action] = (reward + self.discount_factor * next_Q )

        # ì¶œë ¥ ê°?reshape
        target_Q = np.reshape(target_Q, [1, self.action_size])
        # ?¸ê³µ? ê²½ë§??…ë°?´íŠ¸
        self.model.fit(state, target_Q, epochs=1, verbose=0)


if __name__ == "__main__":
    # ?˜ê²½ê³??ì´?„íŠ¸ ?ì„±
    env = Env(render_speed=0.00001)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size)

    scores, episodes = [], []

    for episode in range(201):
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
            next_action = agent.get_action(next_state)

            # ?˜í”Œë¡?ëª¨ë¸ ?™ìŠµ
            agent.train_model(state, action, reward, next_state, next_action, done)
            score += reward
            state = next_state
            time += 1
            if time > 200:
                done = True

            if done:
                # ?í”¼?Œë“œë§ˆë‹¤ ?™ìŠµ ê²°ê³¼ ì¶œë ¥
                print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(
                      episode, score, agent.epsilon))
                scores.append(score)
                episodes.append(episode)

        # 100 ?í”¼?Œë“œë§ˆë‹¤ ëª¨ë¸ ?€??
        if episode % 100 == 0:
            agent.model.save_weights('save_model/model', save_format='tf')
            pylab.plot(episodes, scores, 'b')
            pylab.xlabel("episode")
            pylab.ylabel("score")
            pylab.savefig("./save_graph/graph.png")

    agent.model.save_weights('save_model/model', save_format='tf')

