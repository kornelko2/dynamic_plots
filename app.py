import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import colorsys
import os
import xml.etree.ElementTree as ET

# Sample data
np.random.seed(42)  # For reproducibility
data = {
    'Date': pd.date_range(start='2023-01-01', periods=50, freq='D'),
    'Value1': np.random.randint(10, 150, size=50),
    'Value2': np.random.randint(5, 100, size=50),
    'Value3': np.random.randint(8, 120, size=50),
    'Value4': np.random.randint(3, 90, size=50),
    'Category': ['A', 'B', 'C', 'D'] * 12 + ['A', 'B']
}

# Create DataFrame
df = pd.DataFrame(data)

# Function to get numeric columns
def get_numeric_columns(df):
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

# Function to get non-numeric columns
def get_non_numeric_columns(df):
    return [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col])]

# Function to generate shades of a base color
def generate_shades(base_color, num_shades):
    if num_shades == 1:
        return [base_color]
    base_color_rgb = pc.hex_to_rgb(base_color)
    base_color_hls = colorsys.rgb_to_hls(*[x / 255.0 for x in base_color_rgb])
    shades = []
    for i in range(num_shades):
        lightness = base_color_hls[1] * (1 - (i / (num_shades - 1)) * 0.5)  # Adjusted to 0.5 for more pronounced effect
        rgb = colorsys.hls_to_rgb(base_color_hls[0], lightness, base_color_hls[2])
        shades.append(f'rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})')
    return shades

# Initialize the Dash app
app = dash.Dash(__name__)

# Get column names for dropdown options
column_options = [{'label': col, 'value': col} for col in df.columns]

# Get numeric column names for Y-axis dropdown options
numeric_column_options = [{'label': col, 'value': col} for col in get_numeric_columns(df)]

# Get non-numeric column names for color dropdown options
non_numeric_column_options = [{'label': col, 'value': col} for col in get_non_numeric_columns(df)]
non_numeric_column_options.append({'label': 'None', 'value': 'None'})

# Define the layout of the app
app.layout = html.Div(
    id='main-container',
    children=[
        html.H1(children='Plotly Dash Example'),

        html.Div(children='Dash: A web application framework for Python.', id='subtitle'),

        html.Label('Select X-axis Column:', id='xaxis-label'),
        dcc.Dropdown(
            id='xaxis-column',
            options=column_options,
            value='Date'  # default value
        ),

        html.Label('Select Y-axis Columns:', id='yaxis-label'),
        dcc.Dropdown(
            id='yaxis-columns',
            options=numeric_column_options,
            value=['Value1', 'Value2'],  # default values
            multi=True
        ),

        html.Label('Select Secondary Y-axis Columns:', id='secondary-yaxis-label'),
        dcc.Dropdown(
            id='secondary-yaxis-columns',
            options=numeric_column_options,
            value=[],  # default values
            multi=True
        ),

        html.Label('Select Color Column:', id='color-label'),
        dcc.Dropdown(
            id='color-column',
            options=non_numeric_column_options,
            value='Category'  # default value
        ),

        html.Label('Color Base:', id='color-base-label'),
        dcc.RadioItems(
            id='color-base',
            options=[
                {'label': 'Y-axis Trace Line', 'value': 'yaxis'},
                {'label': 'Category', 'value': 'category'}
            ],
            value='yaxis'  # default value
        ),

        html.Label('Select Chart Type:', id='chart-type-label'),
        dcc.Dropdown(
            id='chart-type',
            options=[
                {'label': 'Line', 'value': 'lines'},
                {'label': 'Bar', 'value': 'bars'},
                {'label': 'Scatter', 'value': 'markers'},
                {'label': 'Histogram', 'value': 'histogram'}
            ],
            value='lines'  # default value
        ),

        html.Button('Update Plot', id='update-button', n_clicks=0),
        

        
        html.Br(),  # Add a break here
        html.Br(),  # Add a break here
        html.Label('Load Saved Settings:', id='load-settings-label'),
        html.Br(),  # Add a break here
        dcc.Dropdown(
            id='load-settings-dropdown',
            options=[],
            value=None
        ),
        html.Button('Load Settings', id='load-settings-button', n_clicks=0),
        html.Button('Update Settings', id='update-settings-button', n_clicks=0),
        html.Br(),  # Add a break here
        html.Br(),  # Add a break here
        html.Label('Enter description to a new save file:', id='description-label'),
        html.Br(),  # Add a break here
        dcc.Input(id='description-input', type='text', value='', placeholder='Enter description'),
        html.Button('Save Settings', id='save-settings-button', n_clicks=0),
        dcc.Graph(
            id='example-graph'
        )
    ]
)

# Function to save settings to an XML file
def save_settings(filename, settings):
    root = ET.Element("settings")
    for key, value in settings.items():
        child = ET.SubElement(root, key)
        child.text = str(value)
    tree = ET.ElementTree(root)
    tree.write(filename)

# Function to load settings from an XML file
def load_settings(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    settings = {child.tag: child.text for child in root}
    return settings

# Function to get list of saved settings files
def get_saved_settings_files():
    if not os.path.exists('./saved_setups'):
        os.makedirs('./saved_setups')
    return [{'label': f, 'value': f} for f in os.listdir('./saved_setups') if f.endswith('.xml')]

# Update the saved settings dropdown options
@app.callback(
    Output('load-settings-dropdown', 'options'),
    [Input('save-settings-button', 'n_clicks')]
)
def update_saved_settings_dropdown(n_clicks):
    return get_saved_settings_files()

# Load settings and apply to controls
@app.callback(
    [Output('xaxis-column', 'value'),
     Output('yaxis-columns', 'value'),
     Output('secondary-yaxis-columns', 'value'),
     Output('color-column', 'value'),
     Output('color-base', 'value'),
     Output('chart-type', 'value')],
    [Input('load-settings-button', 'n_clicks')],
    [State('load-settings-dropdown', 'value')]
)
def load_settings_to_controls(n_clicks, filename):
    if filename is None:
        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update)
    try:
        settings = load_settings(f'./saved_setups/{filename}')
        return (settings['xaxis_column'],
                settings['yaxis_columns'].split(',') if settings['yaxis_columns'] else [],
                settings['secondary_yaxis_columns'].split(',') if settings['secondary_yaxis_columns'] else [],
                settings['color_column'],
                settings['color_base'],
                settings['chart_type'])
    except Exception as e:
        print(f"Error loading settings: {e}")
        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update)

# Save settings when button is clicked
@app.callback(
    Output('save-settings-button', 'n_clicks'),
    [Input('save-settings-button', 'n_clicks')],
    [State('xaxis-column', 'value'),
     State('yaxis-columns', 'value'),
     State('secondary-yaxis-columns', 'value'),
     State('color-column', 'value'),
     State('color-base', 'value'),
     State('chart-type', 'value'),
     State('description-input', 'value')]
)
def save_current_settings(n_clicks, xaxis_column, yaxis_columns, secondary_yaxis_columns, color_column, color_base, chart_type, description):
    if n_clicks > 0:
        settings = {
            'xaxis_column': xaxis_column,
            'yaxis_columns': ','.join(yaxis_columns),
            'secondary_yaxis_columns': ','.join(secondary_yaxis_columns),
            'color_column': color_column,
            'color_base': color_base,
            'chart_type': chart_type
        }
        date_str = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'./saved_setups/{date_str}_{description}.xml'
        save_settings(filename, settings)
    return 0

# Update settings when button is clicked
@app.callback(
    Output('update-settings-button', 'n_clicks'),
    [Input('update-settings-button', 'n_clicks')],
    [State('load-settings-dropdown', 'value'),
     State('xaxis-column', 'value'),
     State('yaxis-columns', 'value'),
     State('secondary-yaxis-columns', 'value'),
     State('color-column', 'value'),
     State('color-base', 'value'),
     State('chart-type', 'value')]
)
def update_current_settings(n_clicks, filename, xaxis_column, yaxis_columns, secondary_yaxis_columns, color_column, color_base, chart_type):
    if n_clicks > 0 and filename:
        settings = {
            'xaxis_column': xaxis_column,
            'yaxis_columns': ','.join(yaxis_columns),
            'secondary_yaxis_columns': ','.join(secondary_yaxis_columns),
            'color_column': color_column,
            'color_base': color_base,
            'chart_type': chart_type
        }
        save_settings(f'./saved_setups/{filename}', settings)
    return 0

# Define callback to update graph
@app.callback(
    Output('example-graph', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State('xaxis-column', 'value'),
     State('yaxis-columns', 'value'),
     State('secondary-yaxis-columns', 'value'),
     State('color-column', 'value'),
     State('color-base', 'value'),
     State('chart-type', 'value')]
)
def update_graph(n_clicks, xaxis_column, yaxis_columns, secondary_yaxis_columns, color_column, color_base, chart_type):
    # Create the figure
    fig = go.Figure()

    # Define base colors for each Y-axis column
    base_colors = ['#FF0000', '#00FF00', '#0000FF', '#A52A2A', '#FFA500', '#800080', '#008080', '#000000', '#FFFF00', '#FFC0CB']

    # Get unique categories
    unique_categories = df[color_column].unique() if color_column != 'None' else [None]

    # Add traces for each Y-axis column
    if color_base == 'yaxis':
        for y_index, y_col in enumerate(yaxis_columns + secondary_yaxis_columns):
            base_color = base_colors[y_index % len(base_colors)]
            shades = generate_shades(base_color, len(unique_categories))
            for i, category in enumerate(unique_categories):
                filtered_df = df if category is None else df[df[color_column] == category]
                trace_name = f"{y_col}_{category}" if category is not None else y_col
                color = shades[i]
                yaxis = 'y2' if y_col in secondary_yaxis_columns else 'y'
                if chart_type == 'bars':
                    fig.add_trace(go.Bar(
                        x=filtered_df[xaxis_column], 
                        y=filtered_df[y_col], 
                        name=trace_name,
                        marker=dict(color=color),
                        yaxis=yaxis
                    ))
                elif chart_type == 'histogram':
                    fig.add_trace(go.Histogram(
                        x=filtered_df[xaxis_column], 
                        y=filtered_df[y_col], 
                        name=trace_name,
                        marker=dict(color=color),
                        yaxis=yaxis
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=filtered_df[xaxis_column], 
                        y=filtered_df[y_col], 
                        mode=chart_type, 
                        name=trace_name,
                        line=dict(color=color, width=2) if chart_type == 'lines' else None,
                        marker=dict(color=color) if chart_type == 'markers' else None,
                        yaxis=yaxis
                    ))
    elif color_base == 'category':
        for i, category in enumerate(unique_categories):
            base_color = base_colors[i % len(base_colors)]
            shades = generate_shades(base_color, len(yaxis_columns) + len(secondary_yaxis_columns))
            for y_index, y_col in enumerate(yaxis_columns + secondary_yaxis_columns):
                filtered_df = df if category is None else df[df[color_column] == category]
                trace_name = f"{y_col}_{category}" if category is not None else y_col
                color = shades[y_index]
                yaxis = 'y2' if y_col in secondary_yaxis_columns else 'y'
                if chart_type == 'bars':
                    fig.add_trace(go.Bar(
                        x=filtered_df[xaxis_column], 
                        y=filtered_df[y_col], 
                        name=trace_name,
                        marker=dict(color=color),
                        yaxis=yaxis
                    ))
                elif chart_type == 'histogram':
                    fig.add_trace(go.Histogram(
                        x=filtered_df[xaxis_column], 
                        y=filtered_df[y_col], 
                        name=trace_name,
                        marker=dict(color=color),
                        yaxis=yaxis
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=filtered_df[xaxis_column], 
                        y=filtered_df[y_col], 
                        mode=chart_type, 
                        name=trace_name,
                        line=dict(color=color, width=2) if chart_type == 'lines' else None,
                        marker=dict(color=color) if chart_type == 'markers' else None,
                        yaxis=yaxis
                    ))

    # Update layout
    fig.update_layout(
        title='Sample Data Timeline', 
        xaxis_title=xaxis_column, 
        yaxis_title='Values',
        template='plotly_white',
        yaxis2=dict(
            title='Secondary Y-axis',
            overlaying='y',
            side='right'
        )
    )

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)