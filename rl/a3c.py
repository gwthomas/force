from multiprocessing import Process, Queue, cpu_count
import lasagne
import numpy as np
import theano
import theano.tensor as T
import time

from gtml.nn.network import Collection, Container
from gtml.nn.opt import Optimizer, squared_error
from gtml.rl.core import Episode, partial_rollout, rollouts, discounted_returns
from gtml.rl.env import vector_var_for_space
from gtml.util.misc import conflattenate, clip_by_global_norm


class ActorCritic(Collection):
    def __init__(self, actor, critic, name):
        self.actor = actor
        self.critic = critic
        super().__init__([actor, critic], name)


def actor_learner(setup_fn, in_q, out_q, tmax, reg_value_fit, reg_entropy):
    ac = setup_fn()
    policy, value_fn = ac.actor, ac.critic
    env = policy.env

    observations_var = policy.get_input_var()
    actions_var = vector_var_for_space(env.action_space)
    log_probs_var = policy.get_log_probs_var(actions_var)
    returns_var = T.fvector()
    advantages_var = T.fvector()
    policy_loss_var = -T.sum(log_probs_var * advantages_var)
    if reg_entropy != 0:
        policy_loss_var = policy_loss_var - reg_entropy * policy.get_entropy_var()
    value_fn_loss_var = squared_error(value_fn.get_output_var().flatten(), returns_var)
    loss_var = policy_loss_var + reg_value_fit * value_fn_loss_var
    grad_vars = T.grad(loss_var, ac.get_param_vars())

    input_vars = [observations_var, actions_var, returns_var, advantages_var]
    output_vars = [policy_loss_var, value_fn_loss_var] + grad_vars
    grad_fn = theano.function(
            inputs=input_vars,
            outputs=output_vars,
            allow_input_downcast=True
    )

    episode = Episode()
    idle = True
    while True:
        # Process all messages from the master
        new_params = None
        while not in_q.empty():
            message = in_q.get()
            if message == 'quit':
                print('Quit')
                return
            elif message == 'pause':
                print('Pause')
                idle = True
            elif message == 'run':
                print('Run')
                idle = False
            else:
                new_params = message

        if idle:
            time.sleep(1)
            continue

        if new_params is not None:
            ac.set_params(new_params)

        # Act for a bit
        steps = partial_rollout(policy, episode, tmax)
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
        outputs = grad_fn(observations, actions, returns, advantages)
        grads = outputs[2:]
        out_q.put((steps, grads))

        if episode.done:
            print(episode.discounted_return)
            episode = Episode()


class A3C:
    def __init__(self, setup_fn, load=False, tmax=20, update_fn=lasagne.updates.adam, num_workers=cpu_count()):
        self.global_ac = setup_fn()
        self.update = self.create_updater(self.global_ac.get_param_vars(), update_fn)

        self.in_q = Queue()
        self.out_qs = []
        self.workers = []
        for _ in range(num_workers):
            q = Queue()
            self.workers.append(Process(target=actor_learner, args=[
                    setup_fn, q, self.in_q, tmax, 0.5, 0.01]))
            self.out_qs.append(q)

        for worker in self.workers:
            worker.start()

    def create_updater(self, param_vars, update_fn):
        grad_vars = [param_var.type() for param_var in param_vars]
        updates = update_fn(grad_vars, param_vars)
        return theano.function(inputs=grad_vars, updates=updates)

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
            self.update(*grads)

            params = self.global_ac.get_params()
            self.global_ac.save_params()
            # Broadcast new params to all workers
            for q in self.out_qs:
                q.put(params)

    def cleanup(self):
        for q in self.out_qs:
            q.put('quit')

        print('Cleaning up workers...')
        for worker in self.workers:
            # worker.join()
            worker.terminate()
        print('Done')
