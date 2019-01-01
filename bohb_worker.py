#!/usr/bin/env python3
from argparse import ArgumentParser
import netifaces
import os
import subprocess as S
from time import sleep

import ConfigSpace as CS
from hpbandster.core.worker import Worker


class SuttonWorker(Worker):
    def compute(self, config_id, config: CS.Configuration, budget: float, **kwargs):
        print(config_id)
        print(config)
        name = f'bohb-cp-{self.run_id}-{config_id[0]}-{config_id[1]}-{config_id[2]}-b{budget}'
        print(name)
        cmdline = ['python3', 'car_pole.py']

        for c, v in config.items():
            cmdline += [f'--{c}', str(v)]

        cmdline += ['--id', name]
        cmdline += ['--iters', str(round(budget / config['num-actors']))]
        cmdline += ['--train', 'baac']
        cmdline += ['--short_name']
        cmdline += ['curriculum']

        print('Launching', *cmdline)

        proc = S.Popen(cmdline, stdout=S.PIPE, stderr=S.PIPE)
        stdout, stderr = proc.communicate()
        stdout = stdout.decode('utf-8')
        stderr = stderr.decode('utf-8')

        loss = None
        bo_result = None
        bo_raw = []
        for line in stdout.split('\n'):
            if line.startswith('BO_RESULT'):
                bo_result = line
                loss = -float(line.split()[1])
            if line.startswith('BO_RAW'):
                bo_raw.append(line)

        os.makedirs('txt_logs', exist_ok=True)
        with open(os.path.join('txt_logs', name + '.stdout'), 'a') as f:
            f.write(stdout)

        with open(os.path.join('txt_logs', name + '.stderr'), 'a') as f:
            f.write(stderr)

        res = { 'loss': loss,
                 'info': {'bo_result': bo_result, 'bo_raw': bo_raw}
               }

        print(res)
        return res

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameter(CS.CategoricalHyperparameter('approximator', ['scaled', 'quantized']))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('tbackup', lower=2, upper=500, log=True))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter('discount', lower=0.9, upper=1, log=True))
#        parser.add_argument('--difficulty', type=float, default=0.001)
#        parser.add_argument('--episodes', type=int, default=10000)

        cs.add_hyperparameter(CS.UniformFloatHyperparameter('lr-policy', lower=0.0001, upper=0.1, log=True))
        cs.add_hyperparameter(CS.UniformFloatHyperparameter('lr-value', lower=0.0001, upper=0.1, log=True))

# !!!        parser.add_argument('--lr', type=float, default=0.01)

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('num-actors', lower=1, upper=50))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('num-layers', lower=2, upper=4))
        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('hidden', lower=5, upper=50))

#        parser.add_argument('--trainer', choices='baac maac'.split())

        cs.add_hyperparameter(CS.UniformFloatHyperparameter('temperature', lower=0.1, upper=10, log=True))

        cs.add_hyperparameter(CS.UniformIntegerHyperparameter('qsteps', lower=10, upper=50))

        return cs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--nameserver', default='127.0.0.1')
    parser.add_argument('--nameserver-port', default=12000)
    parser.add_argument('--iface', default='lo')
    parser.add_argument('--run-id', default='tst')
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--warmstart-from')
    parser.add_argument('role', choices=['standalone', 'master', 'worker'])

    args = parser.parse_args()

    print(args)

    host_addr = netifaces.ifaddresses(args.iface)[2][0]['addr']
    print('host_addr', host_addr)

    if args.role == 'standalone':
        sw = SuttonWorker(args.run_id)
        cfg = SuttonWorker.get_configspace().sample_configuration()

        res = sw.compute((42,43,44), cfg, budget=10000)
        print('worker returned', res)
    elif args.role == 'worker':
        sw = SuttonWorker(args.run_id, args.nameserver, args.nameserver_port, host=host_addr)
        sw.run()
    elif args.role == 'master':
        import logging
        import hpbandster.core.nameserver as hpns
        import hpbandster.core.result as hpres
        from hpbandster.optimizers import BOHB as BOHB

        logging.basicConfig(level=logging.INFO)
        rlogger = hpres.json_result_logger(directory=os.path.join('results', args.run_id), overwrite=True)

        print('Starting NS')
        ns = hpns.NameServer(run_id='ns', host=args.nameserver, port=args.nameserver_port, nic_name=args.iface)
        ns.start()
        sleep(1)

        if args.warmstart_from is not None:
            prev_run = hpres.logged_results_to_HBS_result(args.warmstart_from)
        else:
            prev_run = None


        print('Starting BOHB Master')
        min_b = 10**6
        bo = BOHB(SuttonWorker.get_configspace(), run_id=args.run_id,
                  eta=3, min_budget=1*min_b, max_budget=81 * min_b,
                  nameserver=args.nameserver, nameserver_port=args.nameserver_port, host=host_addr,
                  result_logger=rlogger, previous_result=prev_run
                  )

        print('Starting BOBH run')
        res = bo.run(args.num_runs)

        bo.shutdown(shutdown_workers=True)
        ns.shutdown()

        import pickle
        with open(f'final-result-{args.run_id}.pkl', 'wb') as f:
            pickle.dump(res, f)
