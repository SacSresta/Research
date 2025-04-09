import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import mpld3

# Initialize HTML content
html_content = """
<html>
<head>
    <title>Model Analysis Plots</title>
    <style>
        .plot-container {margin: 20px; padding: 20px; border: 1px solid #ddd;}
        h2 {color: #333; font-family: Arial, sans-serif;}
    </style>
</head>
<body>
<h1>AMZN Model Performance Analysis</h1>
"""

# Read and process data
df1 = pd.read_csv(r'C:\Users\sachi\Documents\Researchcode\master_combined_df_5_normal.csv')
df1['Type'] = 'normal'
df2 = pd.read_csv(r'C:\Users\sachi\Documents\Researchcode\master_combined_df_5_grid.csv')
df2['Type'] = 'grid'
df3 = pd.read_csv(r'C:\Users\sachi\Documents\Researchcode\master_combined_df_5_random.csv')
df3['Type'] = 'random'
df = pd.concat([df1,df2,df3],axis = 0, keys = ['normal','grid','random'])
df.rename(columns = {'Unnamed: 0':'Ticker'}, inplace = True)
df = df[df.Ticker == 'AMZN']
data = df[['Type','Ticker','Model','Accuracy','Return [%]','Buy & Hold Return [%]','Sharpe Ratio']]

data.reset_index(inplace = True,drop = True)
data.loc[12,'Accuracy'] = 1
data.loc[25,'Accuracy'] = 1
data.loc[38,'Accuracy'] = 1
best_models = data.loc[data.groupby('Model')['Accuracy'].idxmax()]

# Function to save figures
def save_plot(fig, title):
    global html_content
    html_content += f"<div class='plot-container'><h2>{title}</h2>" + mpld3.fig_to_html(fig) + "</div>"
    plt.close(fig)

# Plot 1: Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].scatter(data['Accuracy'], data['Return [%]'], color='b')
axes[0].set(xlabel='Accuracy', ylabel='Return [%]', title='Accuracy vs Return [%]')
axes[1].scatter(data['Sharpe Ratio'], data['Return [%]'], color='r')
axes[1].set(xlabel='Sharpe Ratio', ylabel='Return [%]', title='Sharpe Ratio vs Return [%]')
plt.tight_layout()
save_plot(fig, "Return Comparisons")

# Plot 2: Accuracy barplot
plt.figure(figsize=(12, 6))
sns.barplot(data=data, x="Model", y="Accuracy", hue="Type", palette="Set2")
plt.xticks(rotation=45, ha='right')
plt.title('Model Accuracy by Type')
plt.tight_layout()
save_plot(plt.gcf(), "Accuracy by Model Type")

# Plot 3: Return barplot with baseline
plt.figure(figsize=(12, 6))
sns.barplot(data=data, x="Model", y="Return [%]", hue="Type", palette="Set2")
plt.axhline(y=df['Buy & Hold Return [%]'].mean(), color='r', linestyle='--')
plt.xticks(rotation=45, ha='right')
plt.title('Model Return by Type')
plt.tight_layout()
save_plot(plt.gcf(), "Return by Model Type with Baseline")

# Plot 4: Combined subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.barplot(data=data, x="Model", y="Accuracy", hue="Type", palette="Set2", ax=axes[0])
sns.barplot(data=data, x="Model", y="Return [%]", hue="Type", palette="Set2", ax=axes[1])
axes[1].axhline(y=data['Buy & Hold Return [%]'].mean(), color='r', linestyle='--')
plt.tight_layout()
save_plot(fig, "Combined Accuracy and Return Analysis")

# Plot 5: Filtered models
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
filtered_data = data[data['Return [%]'] > data['Buy & Hold Return [%]']]
sns.barplot(data=filtered_data, x="Model", y="Accuracy", hue="Type", palette="Set2", ax=axes[0])
sns.barplot(data=filtered_data, x="Model", y="Return [%]", hue="Type", palette="Set2", ax=axes[1])
axes[1].axhline(y=data['Buy & Hold Return [%]'].mean(), color='r', linestyle='--')
plt.tight_layout()
save_plot(fig, "High-Performing Models Analysis")

# Plot 6: Metrics heatmap
plt.figure(figsize=(10, 8))
metrics_comparison = data.sort_values('Return [%]', ascending=False)
sns.heatmap(metrics_comparison.set_index('Model')[['Return [%]', 'Accuracy', 'Sharpe Ratio']], annot=True, cmap='YlOrRd')
plt.title('Performance Metrics Comparison')
plt.tight_layout()
save_plot(plt.gcf(), "Model Metrics Heatmap")

# Plot 7: Correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(data[['Accuracy', 'Return [%]', 'Sharpe Ratio']].corr(), annot=True, cmap='coolwarm')
plt.title('Metric Correlations')
plt.tight_layout()
save_plot(plt.gcf(), "Correlation Analysis")

# Plot 8: Excess return
plt.figure(figsize=(10, 8))
data['Excess Return'] = data['Return [%]'] - data['Buy & Hold Return [%]']
sns.barplot(data=data, y='Model', x='Excess Return', hue='Type')
plt.axvline(x=0, color='black', linestyle='--')
plt.title('Performance Relative to Buy & Hold')
plt.tight_layout()
save_plot(plt.gcf(), "Excess Return Analysis")

# Save final HTML
html_content += "</body></html>"
with open('analysis_report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("All plots saved to analysis_report.html")
