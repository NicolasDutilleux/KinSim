import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

def create_visualization(csv_input, html_output):
    # Load the data
    df = pd.read_csv(csv_input)
    
    # Simplify names (keeping only the bcXXXX part for the X-axis)
    df['strain_short'] = df['strain'].apply(lambda x: x.split('.')[-1])

    # Create two subplots: one for IPD, one for PW
    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=("IPD Kinetics (Mean ± STD)", "PW Kinetics (Mean ± STD)"),
        vertical_spacing=0.15
    )

    # IPD Bar Chart with Error Bars
    fig.add_trace(
        go.Bar(
            x=df['strain_short'], y=df['mean_ipd'],
            error_y=dict(type='data', array=df['std_ipd'], visible=True),
            name='IPD Mean', marker_color='#2E5A88' # Deep Blue
        ), row=1, col=1
    )

    # PW Bar Chart with Error Bars
    fig.add_trace(
        go.Bar(
            x=df['strain_short'], y=df['mean_pw'],
            error_y=dict(type='data', array=df['std_pw'], visible=True),
            name='PW Mean', marker_color='#A32638' # Deep Red
        ), row=2, col=1
    )

    # Styling (Like a clean music score)
    fig.update_layout(
        height=900, width=1300,
        title_text="KinSim Project: Kinetics Summary by Strain",
        template="plotly_white",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Strain Barcode", tickangle=45)
    fig.update_yaxes(title_text="Signal Intensity")

    # Export to HTML
    fig.write_html(html_output)
    print(f"✅ Visualization saved to {html_output}")

if __name__ == "__main__":
    if(len(sys.argv) > 2):
        create_visualization(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python strain_plotly.py <input_csv> <output_html>")
