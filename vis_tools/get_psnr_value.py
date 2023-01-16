import seaborn as sns
import numpy as np
import tqdm

sns.set_theme(style="darkgrid")
txt_file = r'I:\CodeRep\INRCls\results\vis_residual\logger_wo_residual.log'
f = open(txt_file, 'r')
data_lines = f.readlines()
data_lines = data_lines[4:]
data_epoch = []
for epoch_id in tqdm.tqdm(range(2000)):
    tmp_values = []
    for batch_id in range(19):
        line_id = epoch_id*19+batch_id
        value = float(data_lines[line_id].split(' ')[-1].strip())
        tmp_values.append(value)
    data_epoch.append(np.mean(np.array(tmp_values)))
print(data_lines[line_id])
np.save(txt_file.replace('.log', ''), data_epoch)
