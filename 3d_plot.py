#imports
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.cm as cmx
import matplotlib.colors as colors
#more imports
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import os
import sys
# call with categorize_drugs.py drug_labels.csv categorize_labels.csv user.csv

#parameters?
if len(sys.argv) < 3:
    print("Missing parameters.")
    exit(-1)

#input data
filename = sys.argv[1] #"prediction_results.csv"

#read file
try:
    pred_file = open(filename, "r")
    pred_lines = pred_file.readlines()
except:
    print("Missing input file.")
    exit(-1)

#size
drug_size = int(pred_lines[0].split(',')[0].strip()) #10

#drugs
drugs = [] #['Paracetamol Starch', 'Penicillin Procaine', 'Starch', 'Lactose', 'Amoxicillin', 'Cellulose', 'Vitamin C', 'Quinine', 'Benzyl Penicillin', 'Paracetamol' ]

drug_list = pred_lines[3].split(',')

#loop over drugs and add
for drug in drug_list:
    drugs.append(drug.strip())

#grab output from training
#matrix
m = np.empty([drug_size, drug_size])
#loop over rows
for i in range(0, drug_size):
    #split row
    m_line = pred_lines[4 + i].split(',')

    #set elements
    for j in range(0, drug_size):
        try:
            fl = float(m_line[j].strip())
            m[i,j] = fl
        except:
            continue

# setup the figure and axes
fig = plt.figure(figsize=(8, 8))

# A canvas must be manually attached to the figure
canvas = FigureCanvasAgg(fig)

ax1 = fig.add_subplot(111, projection='3d')
#ax2 = fig.add_subplot(122, projection='3d')

# generate colors
cm = plt.get_cmap('jet')
vv = range(drug_size * drug_size)
cNorm = colors.Normalize(vmin=0, vmax=vv[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
colorVals = [scalarMap.to_rgba(i) for i in range(drug_size * drug_size)]

#normalize
m = ((m.T * 100.0)/m.sum(axis=1)).T

#flatten for plot
mf = m.flatten()

# axis data
_x = np.arange(drug_size)
_y = np.arange(drug_size)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = mf
bottom = np.zeros_like(top)
width = depth = .9

ax1.bar3d(x, y, bottom, width, depth, top, shade=True, color=colorVals)
ax1.set_title('Drug/Distractor accuracy')

ax1.xaxis.set_ticks(_x)
ax1.yaxis.set_ticks(_y)

ax1.set_xticklabels(drugs, fontsize=7, ha='right', va='center', ma='right')
ax1.set_yticklabels(drugs, fontsize=7, ha='left', va='bottom', ma='right')
ax1.set_zlabel('%')

#plt.show()
#save image
fig.savefig(sys.argv[2])
