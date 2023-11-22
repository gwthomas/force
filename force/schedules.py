from force.config import BaseConfig, Configurable, Field


class LinearSchedule(Configurable):
    class Config(BaseConfig):
        initial_value = Field(float)
        final_value = Field(float)
        initial_iters = 0
        linear_iters = Field(int)

    def __call__(self, iteration):
        cfg = self.cfg
        offset = iteration - cfg.initial_iters
        if offset < 0:
            return cfg.initial_value
        elif offset > cfg.linear_iters:
            return cfg.final_value
        linear_frac = offset / cfg.linear_iters
        assert 0 <= linear_frac <= 1
        return cfg.initial_value + linear_frac * (cfg.final_value - cfg.initial_value)