import pygal
from pygal.style import Style
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import to_rgba

### Plot a Bar Graph with Error Bars that represent model performance on the validation dataset ###
# Define model metrics to plot
model_metrics = ['MCC', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'AUPRC']

# Initialize lists that define the values of each model's performance metrics and its standard deviation
chemberta_metrics = [0.834, 0.930, 0.940, 0.962, 0.951, 0.858, 0.930]
chemberta_errors = [0.043, 0.018, 0.020, 0.013, 0.012, 0.050, 0.019]

molgen_metrics = [0.832, 0.929, 0.944, 0.956, 0.949, 0.868, 0.933]
molgen_errors = [0.044, 0.018, 0.021, 0.013, 0.013, 0.052, 0.020]

selfies_ted_metrics = [0.828, 0.927, 0.940, 0.956, 0.948, 0.861, 0.930]
selfies_ted_errors = [0.040, 0.017, 0.018, 0.019, 0.012, 0.044, 0.017]

bartsmiles_metrics = [0.828, 0.928, 0.939, 0.959, 0.949, 0.856, 0.929]
bartsmiles_errors = [0.043, 0.018, 0.022, 0.014, 0.012, 0.057, 0.021]

molformer_metrics = [0.839, 0.932, 0.947, 0.957, 0.952, 0.875, 0.936]
molformer_errors = [0.042, 0.017, 0.022, 0.015, 0.012, 0.056, 0.020]

# Define x-coordinates of bars
x = np.arange(len(model_metrics))

# Define the width of each bar
width=0.15

# Create figure object
fig, ax = plt.subplots(figsize=(18, 4.5))

# Plot bar graphs demonstrating the performance metrics of each model and its standard deviation
ax.bar(x=x, 
       height=chemberta_metrics, 
       width=width, 
       yerr=chemberta_errors,
       linewidth=1.5,
       edgecolor='#0F766E',
       color=to_rgba('#0F766E', 0.3), 
       capsize=4, 
       label='ChemBERTa-2')

ax.bar(x=x-width, 
       height=molgen_metrics, 
       width=width, 
       yerr=molgen_errors,
       linewidth=1.5,
       edgecolor='#D97706',
       color=to_rgba('#D97706', 0.3), 
       capsize=4, 
       label='MolGen')

ax.bar(x=x+width, 
       height=selfies_ted_metrics, 
       width=width, 
       yerr=selfies_ted_errors,
       linewidth=1.5,
       edgecolor='#C026D3',
       color=to_rgba('#C026D3', 0.3), 
       capsize=4, 
       label='SELFIES-TED')

ax.bar(x=x+2*width, 
       height=bartsmiles_metrics, 
       width=width, 
       yerr=bartsmiles_errors,
       linewidth=1.5,
       edgecolor='#334155',
       color=to_rgba('#334155', 0.3), 
       capsize=4, 
       label='BARTSmiles')

ax.bar(x=x-2*width, 
       height=molformer_metrics, 
       width=width, 
       yerr=molformer_errors, 
       linewidth=1.5,
       edgecolor='#3730A3',
       color=to_rgba('#3730A3', 0.3), 
       capsize=4, 
       label='MoLFormer')

# Add formatting and labels
ax.set_ylabel('Scores')
ax.set_xticks(x)
ax.set_xticklabels(model_metrics)
ax.legend(loc='lower right')
plt.show()


### Plot a Bar Graph that shows the proportion of correctly guessed positive cases in the DrugBank Database for all models ###
# Define proportion values for each model
molformer_proportion = 0.909
deepred_bbb_proportion = 0.755
ensemble_bbb_proportion = 0.858 
b3clf_proportion = 0.865
deep_b3_proportion = 0.877

# Define the model names
model_names=['MoLFormer', 'DeePred-BBB', 'EnsembleBBB', 'B3clf', 'Deep-B3']

# Define x-positions of each bar
x = np.arange(5)

# Define the width of each bar
width=0.5

# Create figure object
fig, ax = plt.subplots(figsize=(10, 4.5))

# Plot bar graphs demonstrating the proportion of correctly guessed BBB+ compounds for each model
ax.bar(x[0], 
       height=molformer_proportion, 
       linewidth=1.5,
       edgecolor='#D95F02',
       color=to_rgba('#D95F02', 0.3),
       label='MoLFormer',
       width=width)

ax.bar(x[1], 
       height=deepred_bbb_proportion, 
       linewidth=1.5,
       edgecolor='#0B3B60',
       color=to_rgba('#0B3B60', 0.3),
       label='DeePred-BBB',
       width=width)

ax.bar(x[2], 
       height=ensemble_bbb_proportion,
       linewidth=1.5,
       edgecolor='#E60049',
       color=to_rgba('#E60049', 0.3), 
       label='EnsembleBBB',
       width=width)

ax.bar(x[3], 
       height=b3clf_proportion, 
       linewidth=1.5,
       edgecolor='#047857',
       color=to_rgba('#047857', 0.3),
       label='B3clf',
       width=width)

ax.bar(x[4], 
       height=deep_b3_proportion, 
       linewidth=1.5,
       edgecolor='#7C3AED',
       color=to_rgba('#7C3AED', 0.3),
       label='Deep-B3',
       width=width)

# Add formatting and labels
ax.set_ylabel('Proportion')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend(loc='lower right')
plt.show()


### Plot a Bar Graph that shows the proportion of correctly guessed negatively cases in the DrugBank Database for all models ###
# Define proportion values for each model
molformer_proportion = 0.666
deepred_bbb_proportion = 1
ensemble_bbb_proportion = 1
b3clf_proportion = 1
deep_b3_proportion = 0.666

# Define the model names
model_names=['MoLFormer', 'DeePred-BBB', 'EnsembleBBB', 'B3clf', 'Deep-B3']

# Define x-positions of each bar
x = np.arange(5)

# Define the width of each bar
width=0.5

# Create figure object
fig, ax = plt.subplots(figsize=(10, 4.5))

# Plot bar graphs demonstrating the proportion of correctly guessed BBB- compounds for each model
ax.bar(x[0], 
       height=molformer_proportion, 
       linewidth=1.5,
       edgecolor='#D95F02',
       color=to_rgba('#D95F02', 0.3),
       label='MoLFormer',
       width=width)

ax.bar(x[1], 
       height=deepred_bbb_proportion, 
       linewidth=1.5,
       edgecolor='#0B3B60',
       color=to_rgba('#0B3B60', 0.3),
       label='DeePred-BBB',
       width=width)

ax.bar(x[2], 
       height=ensemble_bbb_proportion,
       linewidth=1.5,
       edgecolor='#E60049',
       color=to_rgba('#E60049', 0.3), 
       label='EnsembleBBB',
       width=width)

ax.bar(x[3], 
       height=b3clf_proportion, 
       linewidth=1.5,
       edgecolor='#047857',
       color=to_rgba('#047857', 0.3),
       label='B3clf',
       width=width)

ax.bar(x[4], 
       height=deep_b3_proportion, 
       linewidth=1.5,
       edgecolor='#7C3AED',
       color=to_rgba('#7C3AED', 0.3),
       label='Deep-B3',
       width=width)

# Add formatting and labels
ax.set_ylabel('Proportion')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend(loc='lower right')
plt.show()


### Plot a Radar Chart of the Model's Performance on the Filtered TITAN-BBB Dataset ###
# Define a custom style for the radar chart
custom_style = Style(
    # Set figure background color
    background='white',
    plot_background='white',

    # Set text color
    foreground='#222222',

    # Set figure lines color
    foreground_strong='#000000',

    # Set opacity of fill and fill line
    opacity='0.1',
    stroke_opacity='1',

    # Set font
    font_family='Arial, sans-serif',
    title_font_size=16,
    label_font_size=9,
    major_label_font_size=9,
    legend_font_size=11,

    # Set legend colors
    colors=('#0B3B60', '#E60049', '#047857', '#7C3AED', '#D95F02'),

    # Disable dotted line configuration for figure
    guide_stroke_dasharray='none',
    major_guide_stroke_dasharray='none',
    
)

# Define custom CSS to thin out the grid lines to 0.4 px
custom_css = (
    'file://style.css',
    'file://graph.css',
    'inline:.axis path, .axis line, .guide path, .guide line { stroke-width: 0.4px !important; }'
)

# Generate a radar chart object with a custom style
filtered_titan_bbb_radar_chart = pygal.Radar(
    # Define dimensions of figure
    width=400,                    
    height=400, 

    # Set fill to true          
    fill=True, 

    # Define range of Radar Chart
    range=(0, 1), 

    # Input custom style and CSS
    style=custom_style,
    css=custom_css,

    # Define the size of the dots on the figure
    dots_size=0.1,

    # Define legend format
    legend_at_bottom=True,        
    legend_at_bottom_columns=3,

    # Define the number of empty padding around the figure
    margin=40,  
)

# Add model label and metrics on filtered TITAN-BBB dataset
# Originally 'MCC', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'AUPRC' but ommitted for figure creation
filtered_titan_bbb_radar_chart.x_labels = [' ', ' ', ' ', ' ', ' ', ' ', ' ']
filtered_titan_bbb_radar_chart.add('DeePred-BBB', [0, 0.439, 0.504, 0.294, 0.371, 0.626, 0.547])
filtered_titan_bbb_radar_chart.add('EnsembleBBB', [0.806, 0.903, 0.932, 0.854, 0.891, 0.945, 0.864])
filtered_titan_bbb_radar_chart.add('B3clf', [0.374, 0.688, 0.661, 0.672, 0.667, 0.701, 0.596])
filtered_titan_bbb_radar_chart.add('Deep-B3', [0.551, 0.778, 0.753, 0.899, 0.819, 0.623, 0.734])
filtered_titan_bbb_radar_chart.add('MoLFormer', [0.551, 0.775, 0.751, 0.876, 0.808, 0.656, 0.725])

# Turn legend off for figure creation purposes 
filtered_titan_bbb_radar_chart.show_legend = False
filtered_titan_bbb_radar_chart.render_to_file('filtered_titan-bbb_results.svg')


### Plot a Radar Chart of the Model's Performance on the Unfiltered TITAN-BBB Dataset ###
# Generate a radar chart object with a custom style
titan_bbb_radar_chart = pygal.Radar(
    # Define dimensions of figure
    width=400,                    
    height=400, 

    # Set fill to true          
    fill=True, 

    # Define range of Radar Chart
    range=(0, 1), 

    # Input custom style and CSS
    style=custom_style,
    css=custom_css,

    # Define the size of the dots on the figure
    dots_size=0.1,

    # Define legend format
    legend_at_bottom=True,        
    legend_at_bottom_columns=3,

    # Define the number of empty padding around the figure
    margin=40,  
)

# Add model label and metrics on unfiltered TITAN-BBB dataset
# Originally 'MCC', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'AUPRC' but ommitted for figure creation
titan_bbb_radar_chart.x_labels = [' ', ' ', ' ', ' ', ' ', ' ', ' ']
titan_bbb_radar_chart.add('DeePred-BBB', [0, 0.439, 0.504, 0.294, 0.371, 0.626, 0.547])
titan_bbb_radar_chart.add('EnsembleBBB', [0.922, 0.961, 0.973, 0.955, 0.964, 0.968, 0.953])
titan_bbb_radar_chart.add('B3clf', [0.434, 0.719, 0.741, 0.738, 0.739, 0.697, 0.688])
titan_bbb_radar_chart.add('Deep-B3', [0.546, 0.776, 0.754, 0.895, 0.818, 0.623, 0.734])
titan_bbb_radar_chart.add('MoLFormer', [0.551, 0.775, 0.751, 0.876, 0.808, 0.656, 0.725])

# Turn legend off for figure creation purposes 
titan_bbb_radar_chart.show_legend = False
titan_bbb_radar_chart.render_to_file('titan-bbb_results.svg')