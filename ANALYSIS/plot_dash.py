import dash
from dash import dcc, html, Input, Output, State,dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import os

path = os.getcwd()
app = dash.Dash(__name__, 
    suppress_callback_exceptions=True,
    update_title=None  # Prevents the "Updating..." message
)
server = app.server

# Define constants
AVAILABLE_TICKERS = ['AAPL', 'SPY', 'NFLX', 'MSFT', 'AMZN', 'AMD','META']
WINDOW_RANGE = list(range(5, 61, 5))

colors = {
    'background': '#F3F6FA',
    'text': '#2C3E50',
    'title': '#34495E',
    'panel': '#FFFFFF'
}

app.layout = html.Div([

    
    # Header
    html.Div([
        html.H1("Stock Model Performance Dashboard",
                style={'textAlign': 'center', 
                      'color': colors['title'],
                      'fontFamily': 'Helvetica',
                      'marginBottom': '30px',
                      'marginTop': '20px'})
    ]),
    
    # Input Controls Panel
    html.Div([
        html.Div([
            html.Label('Select Ticker:', style={'fontSize': '16px', 'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='ticker-input',
                options=[{'label': ticker, 'value': ticker} for ticker in AVAILABLE_TICKERS],
                value='AMZN',
                className='dropdown-ticker',
                style={'width': '200px', 'marginRight': '20px'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '40px'}),
        
        html.Div([
    html.Label('Window Size:', style={'fontSize': '16px', 'fontWeight': 'bold'}),
    dcc.Dropdown(
        id='window-input',
        options=[{'label': f'{window}', 'value': window} for window in WINDOW_RANGE],
        value=5,
        style={
            'width': '200px',
            'marginRight': '20px',
            'borderRadius': '5px'
        }
    ),
], style={'display': 'inline-block', 'marginRight': '40px'}),

        
        html.Button('Update Dashboard', 
                   id='update-button', 
                   n_clicks=0,
                   style={
                       'backgroundColor': '#2C3E50',
                       'color': 'white',
                       'padding': '10px 20px',
                       'borderRadius': '5px',
                       'border': 'none',
                       'cursor': 'pointer',
                       'fontSize': '16px'
                   })
    ],  style={
    'margin': '20px',
    'padding': '20px',
    'backgroundColor': colors['panel'],
    'borderRadius': '10px',
    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
    'display': 'flex',
    'alignItems': 'center',
    'justifyContent': 'center',
    'position': 'sticky',  # Makes it sticky
    'top': '0',           # Sticks to top
    'zIndex': '1000',     # Ensures it stays on top
    'width': '100%'       # Full width
}),
    
    # Content Division for Plots
    html.Div(id='dashboard-content', style={'backgroundColor': colors['background']})
], style={'backgroundColor': colors['background'], 'padding': '20px'})

@app.callback(
    Output('dashboard-content', 'children'),
    [Input('update-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('window-input', 'value')],
    prevent_initial_call=True  # Add this line
)
def update_dashboard(n_clicks, ticker, window):
    df1 = pd.read_csv(f'{path}\\master_combined_df_{window}_normal.csv')
    df1['Type'] = 'normal'
    df2 = pd.read_csv(f'{path}\\master_combined_df_{window}_grid.csv')
    df2['Type'] = 'grid'
    df3 = pd.read_csv(f'{path}\\master_combined_df_{window}_random.csv')
    df3['Type'] = 'random'
    
    df = pd.concat([df1, df2, df3], axis=0, keys=['normal', 'grid', 'random'])
    df.rename(columns={'Unnamed: 0': 'Ticker'}, inplace=True)
    df = df[df.Ticker == ticker]
    data = df[['Start', 'End','Type', 'Ticker', 'Model', 'Accuracy', 'Return [%]', 'Buy & Hold Return [%]', 'Sharpe Ratio','Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration',
       'Avg. Drawdown Duration']]
    
    data.reset_index(inplace=True, drop=True)
    data['Excess Return'] = data['Return [%]'] - data['Buy & Hold Return [%]']
    

    return [
        #Table
        html.Div([
        html.H2("Detailed Results"),
        dash_table.DataTable(
            id='results-table',
            columns=[
                {"name": i, "id": i} for i in data.columns
            ],
            data=data.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'backgroundColor': colors['panel']
            },
            style_header={
                'backgroundColor': colors['title'],
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': colors['background']
                }
            ],
            page_size=13,  # number of rows per page
            sort_action='native',  # enables sorting
            filter_action='native'  # enables filtering
        )
    ], style={
        'margin': '20px',
        'padding': '20px',
        'backgroundColor': colors['panel'],
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
        # Bar Plots
        html.Div([
            html.H2("Model Comparison"),
            dcc.Graph(
                figure=px.bar(
                    data,
                    x="Model",
                    y="Return [%]",
                    color="Type",
                    barmode="group",
                    title=f"{ticker} Return [%] by Type for period {window}"
                ).add_hline(
                    y=data['Buy & Hold Return [%]'].mean(),
                    line_dash="dash",
                    line_color="red"
                )
            ),
            dcc.Graph(
                figure=px.bar(
                    data,
                    x="Model",
                    y="Accuracy",
                    color="Type",
                    barmode="group",
                    title=f"{ticker} Accuracy by Type for period {window}"
                ).add_hline(
                    y=data['Accuracy'].min(),
                    line_dash="dash",
                    line_color="red"
                )
            )
        ]),
        # Scatter Plots
        html.Div([
            html.H2("Return Relationships"),
            dcc.Graph(
                figure=px.scatter(
                    data,
                    x='Accuracy',
                    y='Return [%]',
                    title=f'{ticker} Accuracy vs Return [%]'
                ),
                style={'width': '49%', 'display': 'inline-block'}
            ),
            dcc.Graph(
                figure=px.scatter(
                    data,
                    x='Sharpe Ratio',
                    y='Return [%]',
                    title=f'{ticker} Sharpe Ratio vs Return [%]',
                    color='Type'
                ),
                style={'width': '49%', 'display': 'inline-block'}
            )
        ]),

        

        # Heatmaps and Excess Return
        html.Div([
            html.H2("Advanced Analysis"),
            dcc.Graph(
                figure=px.imshow(
                    data.sort_values('Return [%]', ascending=False)[['Model','Return [%]', 'Accuracy', 'Sharpe Ratio']].set_index('Model'),
                    text_auto=True,
                    title=f"{ticker} Model Metrics Heatmap",
                    aspect='auto'
                )
            ),
            dcc.Graph(
                figure=px.imshow(
                    data[['Accuracy', 'Return [%]', 'Sharpe Ratio']].corr(),
                    text_auto=True,
                    title=f"{ticker} Metric Correlations"
                )
            ),
            dcc.Graph(
                figure=px.bar(
                    data,
                    y="Model",
                    x="Excess Return",
                    color="Type",
                    barmode="group",
                    orientation='h',
                    title=f"{ticker} Excess Return vs Buy & Hold",
                    color_discrete_sequence=["rgb(67, 147, 195)", "rgb(244, 109, 67)", "rgb(116, 196, 118)"]
                ).update_layout(
                    xaxis_title="Excess Return (%)",
                    showlegend=True
                ).add_vline(
                    x=0,
                    line_dash="dash"
                )
            )
        ])
    ]

if __name__ == '__main__':
    app.run(debug=True)
