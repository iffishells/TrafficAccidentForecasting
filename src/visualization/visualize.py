import os

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
import matplotlib.pyplot as plt
from typing import Optional


class DataVisualization:

    def __init__(self):
        self.parent_path_for_saving_html_visualization = os.path.join('..','Datasets_visualization/Visualization/html')
        self.parent_path_for_saving_image_visualization = os.path.join('..','Datasets_visualization','Visualization','image')

        os.makedirs(self.parent_path_for_saving_html_visualization, exist_ok=True)
        os.makedirs(self.parent_path_for_saving_image_visualization, exist_ok=True)

    def Visualize(self,
                  data: pd.DataFrame = None,
                  title: Optional[str] = None,
                  x_label: Optional[str] = None,
                  y_label: Optional[str] = None

                  ):
        # Create a Plotly figure
        fig = go.Figure()

        # Add a trace to the figure
        fig.add_trace(go.Scatter(x=data[x_label], y=data[y_label], name=title))

        # Update trace information and layout
        fig.update_traces(hoverinfo='text+name', mode='lines+markers')
        fig.update_layout(legend=dict(y=0.1, traceorder='reversed', font_size=12))
        fig.update_layout(title=str(title), width=1600)

        # Enable range selector for the x-axis
        fig.update_xaxes(rangeslider_visible=True,
                         rangeselector=dict(buttons=list([dict(step="all")])))

                # Set labels on x-axis and y-axis
        fig.update_xaxes(title_text=f"Date:{x_label}")
        fig.update_yaxes(title_text=f"{y_label}")

        pio.write_image(fig,
                        os.path.join(self.parent_path_for_saving_image_visualization, title),
                        format='png')

        pio.write_html(fig,
                       os.path.join(self.parent_path_for_saving_html_visualization, f'{title}.html'),
                       # format='html'/
                       )

    def __call__(self,
                 df=pd.DataFrame,
                 scatter_plot: bool = False,
                 x_label: Optional[str] = None,
                 y_label: Optional[str] = None,
                 title: Optional[str] = None,


                 ):
        if scatter_plot:
            self.Visualize(data=df,
                           title=title,
                           x_label=x_label,
                           y_label=y_label)
