"""CPU thread limits for one training process and its data-loader workers."""
import os


def training_environment(cpu_threads=1):
    if not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise ValueError('cpu_threads_per_process must be a positive integer')
    env = dict(os.environ)
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[name] = str(cpu_threads)
    env['TOKENIZERS_PARALLELISM'] = 'false'
    return env
