import subprocess
import glob


def get_dims(first, last):
    """
    Returns a list of dimensions:
    [[first^1,first^1],[first^2,first^2],[first^3,first^3],...[last,last]]
    """
    l = []
    i = first
    l.append(i)
    while True:
        i = i * 2
        if i <= last:
            l.append(i)
        else:
            break
    zipped = list(zip(l, l))
    zipped_list = []
    for z in zipped:
        zipped_list.append(list(z))
    return zipped_list


def create_animation(policy_dims=[64, 64]):
    """
    Creates a gif animation with gifski
    """
    policy_dims_str = "_".join([str(x) for x in policy_dims])
    animation_filename = f'animation_{policy_dims_str}'
    path = f'policy_figures/envs:Car1DEnv-v1/ppo/{policy_dims_str}'
    figs = [x.replace(f'{path}/', '') for x in glob.glob(f'{path}/fig*.png')]
    subprocess.Popen(['gifski', '-o', f'{animation_filename}.gif'] + figs, cwd=path)
