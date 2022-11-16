from utils import plot_range_dependency, get_config

exp = 'milp'
plot_range_dependency(exp)
# config = get_config()

# for i in range(1, 29, 1):
#     filepath = config['output_dir'] + f'milp/range/ohe_3_0{i}.npy'
#     plot_bocs(filepath)
