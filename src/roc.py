import numpy as np
import matplotlib.pyplot as plt 

fpr = np.loadtxt('fpr_inc_train_inc_model.txt')
tpr = np.loadtxt('tpr_inc_train_inc_model.txt')
roc_auc = 0.91

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Incremental Model \n (Incremental training set)')
plt.legend(loc="lower right")
plt.show()
