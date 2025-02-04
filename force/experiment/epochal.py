from abc import abstractmethod

import torch

from force.config import Optional
from force.experiment.base import Experiment
from force.nn.module import Module


# Note: the term "epoch" is used loosely here to mean
# "however much training you want to do in between evals".
# It is not referring to a single pass over any particular dataset.


class EpochalExperiment(Experiment):
    class Config(Experiment.Config):
        num_epochs = 100
        checkpoint_period = Optional(int)

    def __init__(self, cfg):
        super().__init__(cfg)
        self._epochs_done = 0

        self._checkpointables: dict[str, Module] = {}
        self._ckpt_dir = self._log_dir / 'checkpoints'

    def get_counters(self) -> dict:
        return {
            'epochs': self._epochs_done
        }

    def pre_epoch(self) -> dict:
        return {}

    def post_epoch(self) -> dict:
        return {}

    @abstractmethod
    def epoch(self) -> dict:
        raise NotImplementedError
    
    def register_checkpointables(self, **kwargs):
        for k, v in kwargs.items():
            assert isinstance(k, str)
            assert isinstance(v, Module)
            assert k not in self._checkpointables
            self._checkpointables[k] = v

    def run(self):
        while self._epochs_done < self.cfg.num_epochs:
            pre_info = self.pre_epoch()
            info = self.epoch()
            self._epochs_done += 1
            post_info = self.post_epoch()

            all_info = {**pre_info, **info, **post_info}

            self.log(f'Completed {self._epochs_done} epochs. Info:')
            for k, v in all_info.items():
                self.log(f'\t{k}: {v}')
                if self._summary_writer is not None:
                    self._summary_writer.add_scalar(k, v, self._epochs_done)
            
            if self.cfg.checkpoint_period is not None and \
               self._epochs_done % self.cfg.checkpoint_period == 0:
                self.save_checkpoint()
    
    def save_checkpoint(self, filename: str | None = None):
        self._ckpt_dir.mkdir(exist_ok=True)

        if filename is None:
            filename = f'ckpt-{self._epochs_done}'
        if '.' not in filename:
            # Add file extension
            filename = filename + '.pt'
        ckpt_path = self._ckpt_dir / filename
        
        state_dict = {
            name: module.state_dict()
            for name, module in self._checkpointables.items()
        }
        torch.save(state_dict, ckpt_path)
        self.log(f'Saved checkpoint to {ckpt_path}')

    def load_checkpoint(self,
                        filename: str | None = None,
                        index: int | None = None):
        assert self.ckpt_dir.is_dir()
        assert not ((index is not None) and (filename is not None)), \
               'Cannot pass both index and filename'

        if filename is None:
            if index is None:
                # Enumerate all identifiable checkpoints and use latest available
                indices = []
                for file in self.ckpt_dir.iterdir():
                    if file.name.startswith('ckpt-') and file.name.endswith('.pt'):
                        # Parse index from filename
                        index_str = file.name[5:-3]
                        indices.append(int(index_str))
                index = max(indices)
            filename = f'ckpt-{index}.pt'
        
        ckpt_path = self._ckpt_dir / filename
        state_dict = torch.load(ckpt_path, weights_only=True)

        # Check that keys match
        expected_keys = set(self._checkpointables.keys())
        actual_keys = set(state_dict.keys())
        assert actual_keys == expected_keys, \
               'Key mismatch when loading checkpoint. ' + \
               f'Got:\n\t{actual_keys}\n\tExpected:{expected_keys}'
        
        # Restore modules' states
        for key in expected_keys:
            self._checkpointables[key].load_state_dict(state_dict[key])
        
        self.log(f'Loaded checkpoint from {ckpt_path}')