import pygal
from pygal.style import Style


# Define a custom style for the radar chart
custom_style = Style(
    # Set background color
    background='white',
    plot_background='white',

    # Set text color
    foreground='#222222',
    foreground_strong='#000000',
    foreground_subtle='#e0e0e0',

    # Set opacity
    opacity='0.25',
    stroke_opacity='1',

    # Set font
    font_family='Arial, sans-serif',
    title_font_size=16,
    label_font_size=10,
    legend_font_size=11,

    # Set legend colors
    colors=('#FF6B6B', '#4ECDC4', '#FFE66D', '#A855F7', '#6B9080'),

    # Disable dotted line configuration
    guide_stroke_dasharray='none',
    major_guide_stroke_dasharray='none',
    
)

# Generate a radar chart object with a custom style
radar_chart = pygal.Radar(
    width=400,                    
    height=400,                   
    fill=True, 
    range=(0, 1), 
    style=custom_style,
    dots_size=0.1,
    stroke_style={'width': 1.5},
    legend_at_bottom=True,        
    legend_at_bottom_columns=3,
    margin=30,  
)

# Add model label and metrics
radar_chart.x_labels = ['MCC', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'Specificity']
radar_chart.add('ChemBERTa', [0.495, 0.745, 0.711, 0.892, 0.791, 0.851, 0.571])
radar_chart.add('MoL-Gen', [0.558, 0.774, 0.732, 0.919, 0.815, 0.863, 0.603])
radar_chart.add('MolFormer', [0.541, 0.769, 0.739, 0.888, 0.807, 0.862, 0.630])
radar_chart.add('BARTSmiles', [0.536, 0.767, 0.736, 0.888, 0.805, 0.871, 0.624])
radar_chart.render_to_file('radar_chart.svg')