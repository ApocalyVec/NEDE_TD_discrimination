import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from load_utils import load_eeg_eye_data

conditions=['eye']
subjects = ['s' + str(i) for i in range(8, 16-1)]
data_root_eeg = '/home/apocalyvec/Dropbox/data/NEDE_TD_discrimination/Data/'
data_root_eye = '/home/apocalyvec/Dropbox/data/NEDE_TD_discrimination/Pupil_Data/'

X_all_eeg, X_all_eye, Y_all, x_train_eeg, x_train_eye, x_test_eeg, x_test_eye, y_train, y_test = load_eeg_eye_data(data_root_eeg, data_root_eye, conditions, subjects)

pca = PCA(n_components=2)
# need to flatten list for PCA
principal_components = pca.fit_transform(X_all_eye[:, :, 0])
# group for every speaker

plt.figure(figsize=(8, 8))
cmap = plt.get_cmap('rainbow')  # rotating color wheel
colors = [cmap(i) for i in np.linspace(0, 1, 2)]
sample_colors = [colors[0] if np.all(x == np.array([1, 0])) else colors[1] for x in Y_all]

plt.scatter(principal_components[:, 0], principal_components[:, 1], color=sample_colors)
plt.title('PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Perform T_SNE #######################################################################################
# for p in range(5, 55, 5):
#     print(p)
p = 25
tsne = TSNE(n_components=2, verbose=1, perplexity=p, n_iter=600)
tsne_results = tsne.fit_transform(X_all_eye[:, :, 0])

plt.figure(figsize=(8, 8))
cmap = plt.get_cmap('rainbow')  # rotating color wheel

colors = [cmap(i) for i in np.linspace(0, 1, 2)]
sample_colors = [colors[0] if np.all(x == np.array([1, 0])) else colors[1] for x in Y_all]

plt.scatter(tsne_results[:, 0], tsne_results[:, 1], color=sample_colors)

plt.title('t_SNE Perplexity=' + str(p))
plt.xlabel('t-SNE 2D 1')
plt.ylabel('t-SNE 2')
plt.show()