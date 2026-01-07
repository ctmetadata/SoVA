import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

labels = [
    'Physi.',
    'Safety',
    'Love&Bel.',
    'Self-Esteem',
    'Self-Actual.'
]

matrix = np.array([
    [0,          1,           1,             0.714286,     0.25        ],
    [-1,         0,           0.432099,      0.627119,     0.622642    ],
    [-1,         -0.432099,   0,             -0.080645,    0.25        ],
    [-0.714286,  -0.627119,   0.080645,      0,            0.222222    ],
    [-0.25,      -0.622642,   -0.25,         -0.222222,    0           ]
])

plt.figure(figsize=(14, 10))

heatmap = sns.heatmap(
    matrix,
    annot=False,
    xticklabels=labels,
    yticklabels=labels,
    cmap='RdBu',
    cbar=True,
    vmin=-1,     
    vmax=1,      
    center=0,    
    linewidths=0.8, 
    linecolor='white'
)

ax_heatmap = plt.gca()
cbar_ax = plt.gcf().axes[-1]

heatmap_bbox = ax_heatmap.get_position()
cbar_ax.set_position([
    heatmap_bbox.x1 + 0.01,
    heatmap_bbox.y0,
    0.02,
    heatmap_bbox.height
])

plt.xticks(
    fontsize=45,
    fontfamily='Times New Roman',
    rotation=45,
    ha='right'
)

plt.yticks(
    fontsize=45,
    fontfamily='Times New Roman',
    rotation=0
)

cbar_ax.tick_params(labelsize=30, width=1.5)
for label in cbar_ax.get_yticklabels():
    label.set_fontfamily('Times New Roman')
cbar_ax.set_yticks([-1, -0.5, 0, 0.5, 1])
cbar_ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])

plt.title("gemma-3-27b-it-RAG", fontsize=50, fontfamily='Times New Roman', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('maslow_blue_orange_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('maslow_blue_orange_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()