import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

x_labels = [
    'Protection', 'Destruction', 'Reproduction', 'Reintegration',
    'Affiliation', 'Rejection', 'Exploration', 'Orientation'
]
y_labels = [
    'Fear', 'Anger', 'Joy', 'Sadness', 'Trust', 'Disgust', 'Anticipation', 'Surprise'
]

matrix = np.array([
    [0.162990,  0,          0,          0,          0.003196,  0,          0.009588,  0.019175],
    [0,          0.095876,  0,          0,          0.003196,  0,          0.003196,  0       ],
    [0,          0,          0.015979,  0.003196,  0.003196,  0,          0,          0.009588],
    [0,          0,          0,          0.012784,  0.003196,  0,          0,          0.003196],
    [0.003196,  0,          0,          0,          0.271649,  0.003196,  0.009588,  0.012784],
    [0,          0,          0.003196,  0,          0.003196,  0.105464,  0.006392,  0.025567],
    [0.003196,  0,          0.003196,  0.003196,  0,          0,          0.089485,  0.015979],
    [0,          0,          0,          0,          0,          0,          0,          0.022371]
])

plt.figure(figsize=(14, 10))

threshold = 0.017
matrix_max = 0.3

boundaries = [
    0,                  
    0.001,              
    0.005,              
    0.01,               
    threshold,          
    0.02,               
    0.03,               
    0.05,               
    0.1,                
    0.2,                
    matrix_max          
]

blues = sns.color_palette("Blues", n_colors=256)
colors = [
    blues[0],     
    blues[30],    
    blues[50],    
    blues[70],    
    blues[110],   
    blues[130],   
    blues[160],   
    blues[190],   
    blues[220],   
    blues[255]    
]

cmap = LinearSegmentedColormap.from_list("custom_blues", colors, N=256)
norm = BoundaryNorm(boundaries, cmap.N)

try:
    heatmap = sns.heatmap(
        matrix,
        annot=False,
        fmt='.4f',
        annot_kws={'fontsize': 10, 'fontweight': 'bold', 'color': '#333'},
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=cmap,
        norm=norm,
        cbar=True,
        cbar_kws={},
        linewidths=1.2,
        linecolor='white'
    )
    
    ax_heatmap = plt.gca()
    cbar = heatmap.collections[0].colorbar
    cbar_ax = cbar.ax
    
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
    
    cbar.set_ticks([0, matrix_max])
    cbar.set_ticklabels(['0.0', '0.3'])
    
    plt.title("gemma-3-27b-it-RAG", fontsize=50, fontfamily='Times New Roman', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('smooth_blues_with_threshold.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('smooth_blues_with_threshold.png', dpi=300, bbox_inches='tight')
    plt.show()
    
except Exception as e:
    print(f"Error during plotting: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()