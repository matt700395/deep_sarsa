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

# κ·Έλ¦¬?μ???μ ?μ???₯μ΄???μ΄?νΈ
class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        # ?ν???¬κΈ°? ?λ???¬κΈ° ?μ
        self.state_size = state_size
        self.action_size = action_size

        # ?₯μ΄???μ΄???λΌλ©ν°
        self.discount_factor = 0.99
        self.lr = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .999
        self.epsilon_min = 0.01
        self.model = DeepSARSA(self.state_size, self.action_size, self.lr)

    # ?μ€λ‘??μ ?μ±?Όλ‘ ?λ ? ν
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # λ¬΄μ?ν??λ°ν
            return random.randrange(self.action_size)
        else:
            # λͺ¨λΈλ‘λ????λ ?°μΆ
            # state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    # <s, a, r, s', a'>???νλ‘λ???λͺ¨λΈ ?λ°?΄νΈ
    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # ANN?μ΅
        target_Q = self.model.predict(state)[0]
        next_Q = self.model.predict(next_state)[0][next_action]

        # ?΄μ¬???ν¨???λ°?΄νΈ ??
        if done:
            target_Q[action] = reward
        else:
            target_Q[action] = (reward + self.discount_factor * next_Q )

        # μΆλ ₯ κ°?reshape
        target_Q = np.reshape(target_Q, [1, self.action_size])
        # ?Έκ³΅? κ²½λ§??λ°?΄νΈ
        self.model.fit(state, target_Q, epochs=1, verbose=0)


if __name__ == "__main__":
    # ?κ²½κ³??μ΄?νΈ ?μ±
    env = Env(render_speed=0.00001)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size)

    scores, episodes = [], []

    for episode in range(201):
        done = False
        score = 0
        # env μ΄κΈ°??
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        time = 0

        while not done:
            # ?μ¬ ?ν??????λ ? ν
            action = agent.get_action(state)

            # ? ν???λ?Όλ‘ ?κ²½?μ ????μ€??μ§ν ???ν ?μ§
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)

            # ?νλ‘?λͺ¨λΈ ?μ΅
            agent.train_model(state, action, reward, next_state, next_action, done)
            score += reward
            state = next_state
            time += 1
            if time > 200:
                done = True

            if done:
                # ?νΌ?λλ§λ€ ?μ΅ κ²°κ³Ό μΆλ ₯
                print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(
                      episode, score, agent.epsilon))
                scores.append(score)
                episodes.append(episode)

        # 100 ?νΌ?λλ§λ€ λͺ¨λΈ ???
        if episode % 100 == 0:
            agent.model.save_weights('save_model/model', save_format='tf')
            pylab.plot(episodes, scores, 'b')
            pylab.xlabel("episode")
            pylab.ylabel("score")
            pylab.savefig("./save_graph/graph.png")

    agent.model.save_weights('save_model/model', save_format='tf')

