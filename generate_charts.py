import matplotlib.pyplot as plt
import numpy as np

# Data from your results
models = ['Logistic Regression', 'XGBoost']
male_fpr = [0.148, 0.185]
female_fpr = [0.000, 0.091]

x = np.arange(len(models))
width = 0.35

# Plot 1: False Positive Rates
fig, ax = plt.subplots(figsize=(7, 5))
bar1 = ax.bar(x - width/2, male_fpr, width, label='Male FPR', color='#2c3e50')
bar2 = ax.bar(x + width/2, female_fpr, width, label='Female FPR', color='#e74c3c')

ax.set_ylabel('False Positive Rate (FPR)')
ax.set_title('False Positive Rate Disparity by Gender and Model')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 0.25)

# Add value labels
ax.bar_label(bar1, padding=3)
ax.bar_label(bar2, padding=3)

plt.tight_layout()
plt.savefig('fpr_disparity.png', dpi=300)
print("Saved fpr_disparity.png")