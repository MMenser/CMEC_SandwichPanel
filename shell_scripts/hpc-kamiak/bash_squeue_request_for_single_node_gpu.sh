# salloc --partition=general-compute --nodes=1 --time=72:00:00 --exclusive --gres=gpu:1 --constraint=V100 --job-name=gpu-learn
# fisbatch --partition=general-compute --nodes=1 --time=72:00:00 --exclusive --gres=gpu:1  --job-name=gpu-learn
# Description
# srun --partition=hawkcpu --nodes=1 --ntasks-per-node=32 --time=1:00:00 --exclusive --pty /bin/bash 
# srun --partition=hawkgpu  --nodes=1 --time=72:00:00 --exclusive --gres=gpu:1  --job-name=coo-gpu --pty /bin/bash
NUM_CPU=6
NUM_GPU=1
PARTITION=kamiak
# NODE="--nodelist=sn14"
# srun --partition=hawkgpu  --nodes=1 --ntasks-per-node=$NUM_CPU --time=48:00:00  --mail-type=all --mail-user=haw621@lehigh.edu --gres=gpu:$NUM_GPU  --job-name=coo-gpu --pty /bin/bash
# srun --partition=$PARTITION --nodes=1 --ntasks-per-node=$NUM_CPU --time=48:00:00  --mail-type=all --mail-user=haifeng.wang@wsu.edu --gres=gpu:$NUM_GPU  --job-name=coo-gpu --constraint=GPU-A100 --pty /bin/bash
idev --partition=$PARTITION \
    --gres=gpu:$NUM_GPU  \
    --nodes=1 --ntasks-per-node=$NUM_CPU --time=10:00:00 \
    --mail-type=all --mail-user=$USER@wsu.edu \
    --job-name=coo-gpu $NODE
