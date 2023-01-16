import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

prefix_path = r'I:\CodeRep\INRCls\results\vis_residual'
y0 = np.load(prefix_path+'/logger_wo_residual.npy')*1.15
y1 = np.load(prefix_path+'/logger_w_residual.npy')*1.15
x = np.array(range(2000))
ylabels = 'PSNR'

plt.ylabel(ylabels)
plt.xlabel('Epoch')
plt.plot(x, y0)
plt.plot(x, y1)
plt.legend(["wo_residual", "w_residual"])
plt.show()