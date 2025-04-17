import dash
from dash import dcc, html, Input, Output, State,dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import os

path = os.getcwd()
path = os.path.join(path,'master_combined_risk_test_date_0024')
print(path)
print(f'{path}\\master_combined_df_10_normal_same_test_date.csv')
app = dash.Dash(__name__, 
    suppress_callback_exceptions=True,
    update_title=None  
)
server = app.server

# Define constants
AVAILABLE_TICKERS = ['AAPL', 'SPY', 'NFLX', 'MSFT', 'AMZN', 'AMD','META']
WINDOW_RANGE = list(range(0, 61, 5))

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
    'position': 'sticky',  
    'top': '0',           
    'zIndex': '1000',     
    'width': '100%'       
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
#master_combined_df_60_grid_same_test_date
def update_dashboard(n_clicks, ticker, window):
    df1 = pd.read_csv(f'{path}\\master_combined_df_{window}_normal_same_test_date.csv')
    df1['Type'] = 'normal'
    df2 = pd.read_csv(f'{path}\\master_combined_df_{window}_grid_same_test_date.csv')
    df2['Type'] = 'grid'
    df3 = pd.read_csv(f'{path}\\master_combined_df_{window}_random_same_test_date.csv')
    df3['Type'] = 'random'
    
    df = pd.concat([df1, df2, df3], axis=0, keys=['normal', 'grid', 'random'])
    df.rename(columns={'Unnamed: 0': 'Ticker'}, inplace=True)
    # Rename columns
    df.rename(columns={
        'Model_x': 'Model',
        'Accuracy_x': 'Accuracy',
        'Confusion Matrix_x': 'Confusion Matrix',
        'Start_x': 'Start',
        'End_x': 'End',
        'Duration_x': 'Duration',
        'Exposure Time [%]_x': 'Exposure Time [%]',
        'Equity Final [$]_x': 'Equity Final [$]',
        'Equity Peak [$]_x': 'Equity Peak [$]',
        'Return [%]_x': 'Return [%]',
        'Buy & Hold Return [%]_x': 'Buy & Hold Return [%]',
        'Return (Ann.) [%]_x': 'Return (Ann.) [%]',
        'Volatility (Ann.) [%]_x': 'Volatility (Ann.) [%]',
        'CAGR [%]_x': 'CAGR [%]',
        'Sharpe Ratio_x': 'Sharpe Ratio',
        'Sortino Ratio_x': 'Sortino Ratio',
        'Calmar Ratio_x': 'Calmar Ratio',
        'Alpha [%]_x': 'Alpha [%]',
        'Beta_x': 'Beta',
        'Max. Drawdown [%]_x': 'Max. Drawdown [%]',
        'Avg. Drawdown [%]_x': 'Avg. Drawdown [%]',
        'Max. Drawdown Duration_x': 'Max. Drawdown Duration',
        'Avg. Drawdown Duration_x': 'Avg. Drawdown Duration',
        '# Trades_x': '# Trades',
        'Win Rate [%]_x': 'Win Rate [%]',
        'Best Trade [%]_x': 'Best Trade [%]',
        'Worst Trade [%]_x': 'Worst Trade [%]',
        'Avg. Trade [%]_x': 'Avg. Trade [%]',
        'Max. Trade Duration_x': 'Max. Trade Duration',
        'Avg. Trade Duration_x': 'Avg. Trade Duration',
        'Profit Factor_x': 'Profit Factor',
        'Expectancy [%]_x': 'Expectancy [%]',
        'SQN_x': 'SQN',
        'Kelly Criterion_x': 'Kelly Criterion',
        '_strategy_x': 'Strategy',
        '_equity_curve_x': 'Equity Curve',
        '_trades_x': 'Trades',
        'Accuracy_y': 'Accuracy (y)',
        'Confusion Matrix_y': 'Confusion Matrix (y)',
        'Start_y': 'Start (y)',
        'End_y': 'End (y)',
        'Duration_y': 'Duration (y)',
        'Exposure Time [%]_y': 'Exposure Time [%] (y)',
        'Equity Final [$]_y': 'Equity Final [$] (y)',
        'Equity Peak [$]_y': 'Equity Peak [$] (y)',
        'Return [%]_y': 'Return [%]_(y)',
        'Buy & Hold Return [%]_y': 'Buy & Hold Return [%] (y)',
        'Return (Ann.) [%]_y': 'Return (Ann.) [%] (y)',
        'Volatility (Ann.) [%]_y': 'Volatility (Ann.) [%] (y)',
        'CAGR [%]_y': 'CAGR [%] (y)',
        'Sharpe Ratio_y': 'Sharpe Ratio (y)',
        'Sortino Ratio_y': 'Sortino Ratio (y)',
        'Calmar Ratio_y': 'Calmar Ratio (y)',
        'Alpha [%]_y': 'Alpha [%] (y)',
        'Beta_y': 'Beta (y)',
        'Max. Drawdown [%]_y': 'Max. Drawdown [%] (y)',
        'Avg. Drawdown [%]_y': 'Avg. Drawdown [%] (y)',
        'Max. Drawdown Duration_y': 'Max. Drawdown Duration (y)',
        'Avg. Drawdown Duration_y': 'Avg. Drawdown Duration (y)',
        '# Trades_y': '# Trades (y)',
        'Win Rate [%]_y': 'Win Rate [%] (y)',
        'Best Trade [%]_y': 'Best Trade [%] (y)',
        'Worst Trade [%]_y': 'Worst Trade [%] (y)',
        'Avg. Trade [%]_y': 'Avg. Trade [%] (y)',
        'Max. Trade Duration_y': 'Max. Trade Duration (y)',
        'Avg. Trade Duration_y': 'Avg. Trade Duration (y)',
        'Profit Factor_y': 'Profit Factor (y)',
        'Expectancy [%]_y': 'Expectancy [%] (y)',
        'SQN_y': 'SQN (y)',
        'Kelly Criterion_y': 'Kelly Criterion (y)',
        '_strategy_y': 'Strategy (y)',
        '_equity_curve_y': 'Equity Curve (y)',
        '_trades_y': 'Trades (y)'
    }, inplace=True)
    df = df[df.Ticker == ticker]
    data = df[['Start', 'End','Type', 'Ticker', 'Model', 'Accuracy', 'Return [%]', 'Buy & Hold Return [%]', 'Sharpe Ratio','Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration',
       'Avg. Drawdown Duration','Return [%]_(y)']]
    
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
            page_size=13, 
            sort_action='native',  
            filter_action='native'  
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
        html.Div([
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
                    annotation_text="Buy & Hold",
                    line_dash="dash",
                    line_color="red"
                )
            )
        ], style={'display': 'inline-block', 'width': '48%', 'padding': '0 10px'}),  # Left chart

        html.Div([
            dcc.Graph(
                figure=px.bar(
                    data,
                    x="Model",
                    y="Return [%]_(y)",
                    color="Type",
                    barmode="group",
                    title=f"{ticker} Return [%] with Risk as 0.024 by Type for period {window}"
                ).add_hline(
                    y=data['Buy & Hold Return [%]'].mean(),
                    annotation_text="Buy & Hold",
                    line_dash="dash",
                    line_color="red"
                )
            )
        ], style={'display': 'inline-block', 'width': '48%', 'padding': '0 10px'}),  # Right chart

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
