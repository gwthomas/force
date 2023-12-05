import torch


def never_terminate(next_state):
    return torch.full([len(next_state)], False, device=next_state.device)

def hopper(next_state):
    height = next_state[:, 0]
    angle = next_state[:, 1]
    not_done = torch.isfinite(next_state).all(axis=-1) \
               * torch.abs(next_state[:,1:] < 100).all(axis=-1) \
               * (height > .7) \
               * (torch.abs(angle) < .2)
    done = ~not_done
    return done

def inverted_pendulum(next_state):
    notdone = torch.isfinite(next_state).all(axis=-1) \
            * (torch.abs(next_state[:,1]) <= .2)
    done = ~notdone
    return done

def inverted_double_pendulum(next_state):
    sin1, cos1 = next_state[:,1], next_state[:,3]
    sin2, cos2 = next_state[:,2], next_state[:,4]
    theta_1 = torch.arctan2(sin1, cos1)
    theta_2 = torch.arctan2(sin2, cos2)
    y = 0.6 * (cos1 + torch.cos(theta_1 + theta_2))
    done = y <= 1
    return done

def walker2d(next_state):
    height = next_state[:, 0]
    angle = next_state[:, 1]
    not_done = (height > 0.8) \
             * (height < 2.0) \
             * (angle > -1.0) \
             * (angle < 1.0)
    done = ~not_done
    return done

def ant(next_state):
    x = next_state[:, 0]
    not_done = torch.isfinite(next_state).all(axis=-1) \
             * (x >= 0.2) \
             * (x <= 1.0)
    done = ~not_done
    return done

def humanoid(next_state):
    z = next_state[:,0]
    done = (z < 1.0) + (z > 2.0)
    return done


TERMINATION_FUNCTIONS = {
    'HalfCheetah': never_terminate,
    'Hopper': hopper,
    'InvertedPendulum': inverted_pendulum,
    'InvertedDoublePendulum': inverted_double_pendulum,
    'Walker2d': walker2d,
    'AntMod': ant,
    'HumanoidMod': humanoid
}