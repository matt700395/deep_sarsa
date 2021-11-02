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

# 그리?�월???�제?�서???�살???�이?�트
class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        # ?�태???�기?� ?�동???�기 ?�의
        self.state_size = state_size
        self.action_size = action_size

        # ?�살???�이???�라메터
        self.discount_factor = 0.99
        self.lr = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .999
        self.epsilon_min = 0.01
        self.model = DeepSARSA(self.state_size, self.action_size, self.lr)

    # ?�실�??�욕 ?�책?�로 ?�동 ?�택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 무작?�행??반환
            return random.randrange(self.action_size)
        else:
            # 모델로�????�동 ?�출
            # state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    # <s, a, r, s', a'>???�플로�???모델 ?�데?�트
    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # ANN?�습
        target_Q = self.model.predict(state)[0]
        next_Q = self.model.predict(next_state)[0][next_action]

        # ?�사???�함???�데?�트 ??
        if done:
            target_Q[action] = reward
        else:
            target_Q[action] = (reward + self.discount_factor * next_Q )

        # 출력 �?reshape
        target_Q = np.reshape(target_Q, [1, self.action_size])
        # ?�공?�경�??�데?�트
        self.model.fit(state, target_Q, epochs=1, verbose=0)


if __name__ == "__main__":
    # ?�경�??�이?�트 ?�성
    env = Env(render_speed=0.00001)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size)

    scores, episodes = [], []

    for episode in range(201):
        done = False
        score = 0
        # env 초기??
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        time = 0

        while not done:
            # ?�재 ?�태???�???�동 ?�택
            action = agent.get_action(state)

            # ?�택???�동?�로 ?�경?�서 ???�?�스??진행 ???�플 ?�집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)

            # ?�플�?모델 ?�습
            agent.train_model(state, action, reward, next_state, next_action, done)
            score += reward
            state = next_state
            time += 1
            if time > 200:
                done = True

            if done:
                # ?�피?�드마다 ?�습 결과 출력
                print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(
                      episode, score, agent.epsilon))
                scores.append(score)
                episodes.append(episode)

        # 100 ?�피?�드마다 모델 ?�??
        if episode % 100 == 0:
            agent.model.save_weights('save_model/model', save_format='tf')
            pylab.plot(episodes, scores, 'b')
            pylab.xlabel("episode")
            pylab.ylabel("score")
            pylab.savefig("./save_graph/graph.png")

    agent.model.save_weights('save_model/model', save_format='tf')

