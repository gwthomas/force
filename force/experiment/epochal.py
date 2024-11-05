from abc import abstractmethod

from force.experiment.base import Experiment


class EpochalExperiment(Experiment):
    class Config(Experiment.Config):
        num_epochs = 100

    def __init__(self, cfg):
        super().__init__(cfg)
        self.epochs_done = 0

    def get_counters(self) -> dict:
        return {
            'epochs': self.epochs_done
        }

    def pre_epoch(self) -> dict:
        return {}

    def post_epoch(self) -> dict:
        return {}

    @abstractmethod
    def epoch(self) -> dict:
        raise NotImplementedError

    def run(self):
        while self.epochs_done < self.cfg.num_epochs:
            pre_info = self.pre_epoch()
            info = self.epoch()
            self.epochs_done += 1
            post_info = self.post_epoch()

            all_info = {**pre_info, **info, **post_info}

            self.log(f'Completed {self.epochs_done} epochs. Info:')
            for k, v in all_info.items():
                self.log(f'\t{k}: {v}')
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar(k, v, self.epochs_done)