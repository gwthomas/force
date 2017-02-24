from threading import Thread
from queue import Queue
import multiprocessing as mp
import lasagne
import numpy as np
import os
import theano
import theano.tensor as T
import time

from gtml.nn.network import Collection, Container
from gtml.nn.opt import Optimizer, squared_error
from gtml.rl.core import Episode, partial_rollout, rollouts, discounted_returns
from gtml.rl.env import vector_var_for_space


class ActorCritic(Collection):
    def __init__(self, actor, critic, name):
        self.actor = actor
        self.critic = critic
        super().__init__([actor, critic], name)


class ActorThread(Thread):
    def __init__(self, master, setup_fn, in_q):
        self.setup_fn = setup_fn
        self.tmax = master.tmax
        self.reg_value_fit = master.reg_value_fit
        self.reg_entropy = master.reg_entropy
        self.render = master.render
        self.out_q = master.in_q
        self.in_q = in_q
        super().__init__(daemon=True)

    def run(self):
        ac = self.setup_fn()
        policy, value_fn = ac.actor, ac.critic
        env = policy.env

        observations_var = policy.get_input_var()
        actions_var = vector_var_for_space(env.action_space)
        log_probs_var = policy.get_log_probs_var(actions_var)
        returns_var = T.fvector()
        advantages_var = T.fvector()
        policy_loss_var = -T.sum(log_probs_var * advantages_var)
        if self.reg_entropy != 0:
            policy_loss_var = policy_loss_var - self.reg_entropy * policy.get_entropy_var()
        value_fn_loss_var = squared_error(value_fn.get_output_var().flatten(), returns_var)
        loss_var = policy_loss_var + self.reg_value_fit * value_fn_loss_var
        grad_vars = T.grad(loss_var, ac.get_param_vars())

        input_vars = [observations_var, actions_var, returns_var, advantages_var]
        output_vars = [policy_loss_var, value_fn_loss_var] + grad_vars
        calc_grads = theano.function(
                inputs=input_vars,
                outputs=output_vars,
                allow_input_downcast=True
        )

        episode = Episode()
        idle = True
        while True:
            # Process all messages from the master
            new_params = None
            while not self.in_q.empty():
                message = self.in_q.get()
                if message == 'quit':
                    return
                elif message == 'pause':
                    print('Pausing worker', threading.current_thread())
                    idle = True
                elif message == 'run':
                    idle = False
                else:
                    new_params = message

            if new_params is not None:
                ac.set_params(new_params)

            if idle:
                time.sleep(1)
                continue

            # Act for a bit
            steps = partial_rollout(policy, episode, self.tmax, render=self.render)
            if episode.done:
                R = 0
                observations = episode.observations[-steps:]
            else:
                R = value_fn([episode.latest_observation()])[0]
                observations = episode.observations[-(steps+1):-1]
            actions = episode.actions[-steps:]
            rewards = episode.rewards[-steps:]
            returns = np.zeros(steps)
            for i in range(steps-1, -1, -1):
                R = rewards[i] + env.discount * R
                returns[i] = R

            advantages = returns - value_fn(observations)
            outputs = calc_grads(observations, actions, returns, advantages)
            grads = outputs[2:]
            self.out_q.put((steps, grads))

            if episode.done:
                print(episode.discounted_return)
                episode = Episode()


class A3C(Optimizer):
    def __init__(self, setup_fn, tmax=20, reg_value_fit=0.5, reg_entropy=0.01,
                update_fn=lasagne.updates.adam, num_workers=os.cpu_count(), render=False):
        self.global_ac = setup_fn()
        self.tmax = tmax
        self.reg_value_fit = reg_value_fit
        self.reg_entropy = reg_entropy
        self.render = render

        param_vars = self.global_ac.get_param_vars()
        input_vars = [param_var.type() for param_var in param_vars]
        updates = update_fn(input_vars, param_vars)
        super().__init__(input_vars, updates)

        self.in_q = mp.Queue()
        self.out_qs = []
        self.workers = []
        for _ in range(num_workers):
            q = mp.Queue()
            self.workers.append(ActorThread(self, setup_fn, q))
            self.out_qs.append(q)

        for worker in self.workers:
            worker.start()

    def run(self, Tmax):
        params = self.global_ac.get_params()
        for q in self.out_qs:
            q.put(params)
            q.put('run')

        T = 0
        while T < Tmax:
            # Process a message from the workers
            message = self.in_q.get()
            steps, grads = message
            T += steps
            self.step(*grads)

            params = self.global_ac.get_params()
            self.global_ac.save_params()
            # Broadcast new params to all workers
            for q in self.out_qs:
                q.put(params)

        for q in self.out_qs:
            q.put('pause')

    def cleanup(self):
        for q in self.out_qs:
            q.put('quit')

        print('Cleaning up workers...')
        for worker in self.workers:
            worker.join()
        print('Done')


# class ActorThread(Thread):
#     def __init__(self, master, setup_fn, in_q):
#         self.master = master
#         self.setup_fn = setup_fn
#         self.in_q = in_q
#         super().__init__(daemon=True)
#
#     def run(self):
#         master = self.master
#         global_ac = master.global_ac
#         ac = self.setup_fn()
#         policy, value_fn = ac.actor, ac.critic
#         env = policy.env
#
#         observations_var = policy.get_input_var()
#         actions_var = vector_var_for_space(env.action_space)
#         log_probs_var = policy.get_log_probs_var(actions_var)
#         returns_var = T.fvector()
#         advantages_var = T.fvector()
#         policy_loss_var = -T.sum(log_probs_var * advantages_var)
#         if master.reg_entropy != 0:
#             policy_loss_var = policy_loss_var - master.reg_entropy * policy.get_entropy_var()
#         value_fn_loss_var = squared_error(value_fn.get_output_var().flatten(), returns_var)
#         loss_var = policy_loss_var + master.reg_value_fit * value_fn_loss_var
#         grad_vars = T.grad(loss_var, ac.get_param_vars())
#
#         input_vars = [observations_var, actions_var, returns_var, advantages_var]
#         output_vars = [policy_loss_var, value_fn_loss_var] + grad_vars
#         calc_grads = theano.function(
#                 inputs=input_vars,
#                 outputs=output_vars,
#                 allow_input_downcast=True
#         )
#
#         copy_params = theano.function(inputs=[], outputs=[],
#                 updates=list(zip(ac.get_param_vars(), global_ac.get_param_vars()))
#         )
#
#         episode = Episode()
#         idle = True
#         while True:
#             # Process all messages from the master
#             update = False
#             while not self.in_q.empty():
#                 message = self.in_q.get()
#                 if message == 'quit':
#                     return
#                 elif message == 'pause':
#                     print('Pausing worker', threading.current_thread())
#                     idle = True
#                 elif message == 'run':
#                     idle = False
#                     update = True
#                 elif message == 'update':
#                     update = True
#                 else:
#                     raise RuntimeError('Invalid message:' + str(message))
#
#             if idle:
#                 time.sleep(1)
#                 continue
#
#             if update:
#                 copy_params()
#
#             # Act for a bit
#             steps = partial_rollout(policy, episode, master.tmax, render=master.render)
#             if episode.done:
#                 R = 0
#                 observations = episode.observations[-steps:]
#             else:
#                 R = value_fn([episode.latest_observation()])[0]
#                 observations = episode.observations[-(steps+1):-1]
#             actions = episode.actions[-steps:]
#             rewards = episode.rewards[-steps:]
#             returns = np.zeros(steps)
#             for i in range(steps-1, -1, -1):
#                 R = rewards[i] + env.discount * R
#                 returns[i] = R
#
#             advantages = returns - value_fn(observations)
#             outputs = calc_grads(observations, actions, returns, advantages)
#             grads = outputs[2:]
#             self.master.in_q.put((steps, grads))
#
#             if episode.done:
#                 print(episode.discounted_return)
#                 episode = Episode()
#
#
# class A3C(Optimizer):
#     def __init__(self, setup_fn, tmax=20, reg_value_fit=0.5, reg_entropy=0.01,
#                 update_fn=lasagne.updates.adam, num_workers=os.cpu_count(), render=False):
#         self.global_ac = setup_fn()
#         self.tmax = tmax
#         self.reg_value_fit = reg_value_fit
#         self.reg_entropy = reg_entropy
#         self.render = render
#
#         param_vars = self.global_ac.get_param_vars()
#         input_vars = [param_var.type() for param_var in param_vars]
#         updates = update_fn(input_vars, param_vars)
#         super().__init__(input_vars, updates)
#
#         self.in_q = mp.Queue()
#         self.out_qs = []
#         self.workers = []
#         for _ in range(num_workers):
#             q = Queue()
#             self.workers.append(ActorThread(self, setup_fn, q))
#             self.out_qs.append(q)
#
#         for worker in self.workers:
#             worker.start()
#
#     def run(self, Tmax):
#         for q in self.out_qs:
#             q.put('run')
#
#         T = 0
#         while T < Tmax:
#             # Process a message from the workers
#             message = self.in_q.get()
#             steps, grads = message
#             T += steps
#             print(T)
#             self.step(*grads)
#             self.global_ac.save_params()
#
#             # Let workers know there are new parameters
#             for q in self.out_qs:
#                 q.put('update')
#
#         for q in self.out_qs:
#             q.put('pause')
#
#     def cleanup(self):
#         for q in self.out_qs:
#             q.put('quit')
#
#         print('Cleaning up workers...')
#         for worker in self.workers:
#             worker.join()
#         print('Done')
