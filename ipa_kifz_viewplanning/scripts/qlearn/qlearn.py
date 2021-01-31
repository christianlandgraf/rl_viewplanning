'''
Except for some smaller adjustments, this script is heavily based on:

https://bitbucket.org/theconstructcore/openai_examples_projects/src/master/turtle2_openai_ros_example/scripts/qlearn.py

'''
import random


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha  # discount constant
        self.gamma = gamma  # discount factor
        self.actions = actions

    def initQ(self, states, actions):
        for state in states:
            for action in actions:
                self.q[(state, action)] = 0.1  #0.0001

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            #self.q[(state, action)] = reward
            pass
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, previous_states, return_q=False):
        self.unique_actions = [
            a for a in self.actions if a not in previous_states
        ]

        q = [self.getQ(state, a) for a in self.unique_actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q)
            mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [
                q[i] + random.random() * mag - .5 * mag
                for i in range(len(self.unique_actions))
            ]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.unique_actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.unique_actions[i]
        if return_q:  # if they want it, give it!
            return action, q
        previous_states.append(action)

        return action, previous_states

    def learn(self, state_0, action, reward, state_1):
        maxqnew = max([self.getQ(state_1, a) for a in self.actions])
        self.learnQ(state_0, action, reward, reward + self.gamma * maxqnew)
