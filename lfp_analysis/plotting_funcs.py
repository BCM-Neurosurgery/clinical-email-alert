import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import time as dttime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TIME_INDEX = [dttime(i // 60, i % 60) for i in range(0, 1440, 10)]

def plot_daily_r2(pt, df):

    colors = np.array(['gold', 'r', 'orange', 'b', 'gray'])
    pt_df = df.query('pt_id==@pt and (lead_location=="VC/VS" or lead_location=="OTHER")')
    pt_df_head = pt_df.groupby(pd.Grouper(key='CT_timestamp', freq='D')).head(1)

    plt.figure(figsize=(15,3))
    plt.scatter(pt_df_head['days_since_dbs'], pt_df_head['lfp_left_day_r2_OvER'], s=10, c=colors[pt_df_head['state_label']])

    plt.grid()
    plt.gca().set(xlabel='Days since DBS', ylabel='Daily R2', title=f'{pt_df.loc[pt_df.index[0], "pt_id"]}')
    plt.gca().set_xticks(plt.gca().get_xticks())
    plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=45, ha='right')
    plt.show()
    return

def plot_lfp_heatmap(pt, df):
    pt_df = df.query('pt_id==@pt and (lead_location=="VC/VS" or lead_location=="OTHER")')
    days = pt_df['days_since_dbs'].drop_duplicates().dropna()
    pt_lfp = pd.DataFrame(columns = days, index = [dttime(i // 60, i % 60) for i in range(0, 1440, 10)])
    for day in days:
        day_df = pt_df[pt_df['days_since_dbs'] == day]
        day_df.loc[:,'time_bin'] = day_df['time_bin'] - pd.Timedelta(6, unit='h')
        day_df['time_bin'] = day_df['time_bin'].dt.time
        for time_bin, value in day_df[['time_bin', 'lfp_left_z_scored_OvER']].values:
            if time_bin in pt_lfp.index:
                pt_lfp.loc[time_bin, day] = value

    pt_lfp = pt_lfp.astype(float)
    fig, axs = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(len(pt_lfp.columns)*1, 7))
    p = axs.imshow(pt_lfp, aspect=1.2, cmap='jet')
    cbar = plt.colorbar(p, ax=axs)
    cbar.set_label('9 Hz LFP (z-scored)', rotation=270)

    plt.xlabel('Day')
    plt.ylabel('Time of Day')
    #plt.xticks(np.arange(0, len(pt_lfp.columns), len(pt_lfp.columns) // 10), pt_lfp.columns[::11])
    plt.yticks(np.arange(0, len(pt_lfp.index), 71), ['0:00', '12:00', '24:00'])
    plt.title(pt)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_data(patient, pt_info, df, hemisphere='left'):
    model = 'OvER'

    pt_df = df.query('pt_id == @patient')
    days = pt_df.groupby('days_since_dbs').head(1)['days_since_dbs']
    linAR_R2 = pt_df.groupby('days_since_dbs').head(1)[f'lfp_{hemisphere}_day_r2_{model}']

    # Identify discontinuities in the days array
    start_index = np.where(np.diff(days) > 7)[0] + 1
    start_index = np.concatenate(([0], start_index, [len(days)]))

    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=4,
        row_heights=[0.5, 0.5],
        column_widths=[0.3, 0.35, 0.35, 0.1],
        specs=[[{"colspan": 4}, None, None, None],
                [{"colspan": 3}, None, None, {"colspan": 1}]],
        subplot_titles=("Full Time-Domain Plot",
                        "Linear AR R² Over Time", "Linear AR R² Violin Plot"))

    # Set plot aesthetics
    title_font_color = '#2e2e2e'
    axis_title_font_color = '#2e2e2e'
    axis_line_color = '#2e2e2e'
    plot_bgcolor = 'rgba(240, 240, 240, 1)'
    paper_bgcolor = 'rgba(240, 240, 240, 1)'
    c_preDBS = 'rgba(255, 215, 0, 0.5)'
    c_responder = 'rgba(0, 0, 255, 1)'
    c_disinhibited = '#ff0000'
    c_nonresponder = 'rgba(255, 185, 0, 1)'
    c_dots = 'rgba(128, 128, 128, 0.5)'
    c_linAR = 'rgba(51, 160, 44, 1)'
    c_OG = 'rgba(128, 128, 128, 0.7)'
    sz = 5

    fig.update_layout(
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        annotations=[dict(
            text='',
            xref='paper',
            yref='paper',
            x=0,
            y=1,
            showarrow=False,
            font=dict(
                size=20,
                color=title_font_color,
                family="Helvetica"
            )
        )]
    )
    
    # Plot Full Time-Domain Plot
    for i in range(len(start_index) - 1):
        segment_days = np.ravel(days[start_index[i]+1:start_index[i+1]])
        segment_OG = pt_df.query('days_since_dbs in @segment_days')[f'lfp_{hemisphere}_z_scored_{model}']
        segment_linAR = pt_df.query('days_since_dbs in @segment_days')[f'lfp_{hemisphere}_preds_{model}']
        segment_times = pt_df.query('days_since_dbs in @segment_days')['CT_timestamp']

        mask = ~np.isnan(segment_times) & ~np.isnan(segment_OG)
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            segments = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
            
            for segment in segments:
                if len(segment) > 0:
                    fig.add_trace(go.Scatter(
                        x=segment_times.values[segment],
                        y=segment_OG.values[segment],
                        mode='lines',
                        line=dict(color=c_OG, width=1),
                        showlegend=False
                    ), row=1, col=1)

        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) > 0:
            segments = np.split(valid_indices, np.where(np.diff(valid_indices) != 1)[0] + 1)
            
            for segment in segments:
                if len(segment) > 0:
                    fig.add_trace(go.Scatter(
                        x=segment_times.values[segment],
                        y=segment_linAR.values[segment],
                        mode='lines',
                        line=dict(color=c_linAR, width=1),
                        showlegend=False
                    ), row=1, col=1)

    fig.add_vline(x=pt_info['dbs_date'], line_width=5, line_dash="dash", line_color="hotpink", row=1, col=1)

    fig.update_yaxes(title_text="9 Hz LFP (mV)", row=1, col=1, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(title_text="Date", row=1, col=1, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)

     # Linear AR R² Over Time
    for i in range(len(start_index) - 1):
        segment_days = days.values[start_index[i]+1:start_index[i+1]]
        segment_linAR_R2 = linAR_R2.values[start_index[i]+1:start_index[i+1]]
        
        # Plot the dots
        fig.add_trace(go.Scatter(
            x=segment_days,
            y=segment_linAR_R2,
            mode='markers',
            marker=dict(color=c_dots, size=sz),
            showlegend=False
        ), row=2, col=1)
        
    color_dict = {0: c_preDBS, 1: c_disinhibited, 2: c_nonresponder, 3: c_responder, 4: c_dots}
    fig.add_vline(x=0, line_width=5, line_dash="dash", line_color="hotpink", row=2, col=1)
    
    if patient[0] == 'B' or patient[0] == 'A':
        fig.add_hline(y = 0.3493491960462022, line_width=2, line_dash="dash", line_color="black", row=2, col=1)

    for color in color_dict.keys():
        if color not in pt_df['state_label']:
            continue    
        state_df = pt_df.query('state_label == @color')
        state_days = state_df.groupby('days_since_dbs').head(1)['days_since_dbs']
        state_r2 = state_df.groupby('days_since_dbs').head(1)[f'lfp_{hemisphere}_day_r2_{model}']

        fig.add_trace(go.Scatter(
            x=state_days,
            y=state_r2.rolling(window=5, min_periods=1).mean(),
            mode='lines',
            line=dict(color=color_dict[color]),
            showlegend=False
        ), row=2, col=1)

    fig.update_yaxes(title_text="Linear AR R²", range=(-0.5, 1), row=2, col=1, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(title_text="Days since DBS activation", row=2, col=1, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)

    # Linear AR R² Violin Plot
    VIOLIN_WIDTH = 7.0
    fig.add_trace(go.Violin(
            y=pt_df.query('state_label == 0').groupby('days_since_dbs').head(1)[f'lfp_{hemisphere}_day_r2_{model}'],
            name='', 
            side='negative', 
            line_color=c_preDBS, 
            fillcolor=c_preDBS,
            showlegend=False,
            width=VIOLIN_WIDTH,
            meanline_visible=True, 
            meanline=dict(color='black', width=2)
        ), row=2, col=4)

    if pt_info['response_status'] == 1:
        fig.add_trace(go.Violin(
            y=pt_df.query("days_since_dbs >= @pt_info['response_date']").groupby('days_since_dbs').head(1)[f'lfp_{hemisphere}_day_r2_{model}'],  
            side='positive', 
            line_color=c_responder, 
            fillcolor=c_responder,
            showlegend=False,
            width=VIOLIN_WIDTH,
            meanline_visible=True, 
            meanline=dict(color='white', width=2)
        ), row=2, col=4)
    else:
        fig.add_trace(go.Violin(
            y=pt_df.query('days_since_dbs > 0').groupby('days_since_dbs').head(1)[f'lfp_{hemisphere}_day_r2_{model}'], 
            side='positive', 
            line_color=c_nonresponder, 
            fillcolor=c_nonresponder,
            showlegend=False,
            width=VIOLIN_WIDTH,
            meanline_visible=True, 
            meanline=dict(color='black', width=2)
        ), row=2, col=4)

    fig.update_yaxes(range=(-0.5, 1), row=2, col=4, tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)
    fig.update_xaxes(tickfont=dict(color=axis_title_font_color), titlefont=dict(color=axis_title_font_color), showline=True, linecolor=axis_line_color)

    fig.update_layout(
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig