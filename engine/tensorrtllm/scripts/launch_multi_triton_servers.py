import argparse
import subprocess
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument('--tritonserver', type=str,
                        default='/opt/tritonserver/bin/tritonserver')
    parser.add_argument('--model_repos', nargs='+', type=str,
                        help='List of model repositories', required=True)
    return parser.parse_args()

def get_cmd(world_size, tritonserver, model_repos):
    cmd = 'mpirun --allow-run-as-root '
    for i in range(world_size):
        for model_repo in model_repos:
            cmd += f' -n 1 {tritonserver} --model-repository={model_repo} --disable-auto-complete-config --backend-config=python,shm-region-prefix-name=prefix{i} : '
    cmd += '&'
    return cmd

if __name__ == '__main__':
    args = parse_arguments()
    cmd = get_cmd(args.world_size, args.tritonserver, args.model_repos)
    subprocess.call(cmd, shell=True)
