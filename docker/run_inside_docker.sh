export PATH="/home/jiale/anaconda3/bin:$PATH"
conda init
source ~/.bashrc
conda activate softgym
cd /home/jiale/softgym/
export PYTHONPATH="${PYTHONPATH}:/${PWD}"
source ./prepare_1.0.sh
# source ./compile_1.0.sh

# python examples/random_env.py --env_name PassWater