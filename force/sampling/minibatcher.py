import queue

import torch
import torch.multiprocessing as mp


# Helper function
def _create_buffer(proto, batch_size, share_memory):
    buffer = {}
    for k, v in proto.items():
        buf = torch.zeros(batch_size, *v.shape, dtype=v.dtype, device=v.device)
        if share_memory:
            buf.share_memory_()
        buffer[k] = buf
    return buffer


# Main routine of worker processes
def _sampler_main(proc_index, sampler, buffer, write_q, written_q, running_flag):
    while running_flag.value:
        # This design (get with timeout and ignore exception) allows
        # the loop to terminate when running_flag is set to false
        try:
            fill_index = write_q.get(block=True, timeout=1)
        except queue.Empty:
            continue

        # Generate a sample and write to buffer
        batch = sampler.sample()
        for k, v in batch.items():
            buffer[k][fill_index] = v

        # Let the main process know which index was written to
        written_q.put((proc_index, fill_index))


class Minibatcher:
    """
    Minibatcher uses parallelism across processes for mini-batch sampling.
    Each process has its own batch-sized buffer in which to write samples.
    The original process sends (via process-specific queues) indices to fill.
    Each worker process waits for these messages, samples, writes the sample
    to the corresponding index in its own buffer, and then sends a message back
    to the main process, which aggregates the results into a batch.

    Notes:
     * No locking is needed. Each worker writes to its own buffer. The only
       race condition would be the main process reading from a buffer that is
       being written to by a worker, but such reading/writing will never happen
       at the same index. This is because the worker will not write to an index
       until the main process requests it, but the main process won't request
       an index be refilled until after reading from it.

     * Messages consist of just 1-2 integers. PyTorch tensors are never sent
       via queues, but rather are copied to and from persistent buffers which
       reside in shared memory.
    """

    def __init__(self, samplers, batch_size):
        self.batch_size = batch_size

        # Take a sample to see its size. Assume all samplers return same shape
        proto_sample = samplers[0].sample()
        assert isinstance(proto_sample, dict), 'Currently only dict samples are supported'

        # A boolean flag to communicate termination to subprocesses
        self.running_flag = mp.Value('b', True)

        # Buffer where returned sample is written
        self.out_buf = _create_buffer(proto_sample, batch_size, False)

        # Queue by which worker processes communicate to the main process the
        # index to which they've just written
        self.written_q = mp.SimpleQueue()

        self.proc_bufs = []
        self.write_qs = []
        self.processes = []
        for proc_index, sampler in enumerate(samplers):
            # Each process has its own buffer to write samples to.
            # Memory is shared so that data can be copied from these
            # process-specific buffers to output buffer
            buffer = _create_buffer(proto_sample, batch_size, True)

            # Queue by which the main process communicates to worker processes
            # which indices should be filled with samples.
            # Initially, all indices should be filled.
            write_q = mp.Queue()
            for i in range(batch_size):
                write_q.put(i)

            # Create and start process
            proc = mp.Process(target=_sampler_main, args=(
                proc_index, sampler, buffer, write_q, self.written_q, self.running_flag
            ))
            proc.start()
            self.proc_bufs.append(buffer)
            self.write_qs.append(write_q)
            self.processes.append(proc)

    def sample(self):
        for i in range(self.batch_size):
            # Get from queue to determine where to find sample
            proc_index, fill_index = self.written_q.get()

            # Copy from process-specific buffer to output buffer
            proc_buf = self.proc_bufs[proc_index]
            for k in self.out_buf.keys():
                self.out_buf[k][i] = proc_buf[k][fill_index]

            # Let worker process know to write a new sample to the index
            self.write_qs[proc_index].put(fill_index)
        return self.out_buf

    def stop(self):
        # Setting this to true will end the infinite loop in the workers
        self.running_flag.value = False

        # Wait for them to terminate
        for proc in self.processes:
            proc.join()

        # Close queues
        for write_q in self.write_qs:
            write_q.close()