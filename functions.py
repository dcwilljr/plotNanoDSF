def read_data(datafile_name):
    import pandas as pd
    import numpy as np

    if datafile_name[-3:] == 'csv':
        data_list = pd.read_csv(datafile_name)
        data = np.array(data_list[1:]).astype(float)
    elif datafile_name[-4:] == 'xlsx':
        data_list = pd.read_excel(datafile_name, sheet_name='Ratio')
        data = np.array(data_list[1:]).astype(float)
    else:
        exit("Data file needs to be either 'csv' or 'xlsx'")
    num_samples = len(data_list.iloc[0].values.flatten().tolist())/3
    return data, num_samples

def get_names(exp_names_list):
    names = exp_names_list.split("\n")
    return names

def make_data_df(names, data,
                rl_win = 3,
                ord = 2):
    from scipy.signal import find_peaks
    import derivative
    import pandas as pd
    sg = derivative.SavitzkyGolay(left=rl_win, right=rl_win, order=ord, periodic=False)
    data_df = pd.DataFrame()
    Tm_dict = {}

    for i in range(len(names)):
        x = data[:,(i*3) + 1]
        y = data[:,(i*3) + 2]
        dy = sg.d(y,x)
        if (-1.0*dy).max() > dy.max():
            dy = -1.0*dy
        
        peaks, _ = find_peaks(dy, height = dy.max()*0.5, distance = len(x)*0.05)
        if peaks.size != 0:
            Tm = x[peaks[dy[peaks].argmax()]]
        Tm_dict[names[i] + '_Tm'] = Tm
        Tm_dict[names[i] + '_peaks'] = peaks
        data_df[names[i],'x'] = x
        data_df[names[i],'y'] = y
        data_df[names[i],'dy'] = dy
    return data_df, Tm_dict

def make_blank_overlay(overlay_list,
                       bgcol = 'white'):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.)

    for overlay_name in overlay_list:
        fig.add_trace(go.Scatter(x = [],y=[], showlegend = False,
                                    mode = "lines",
                                    line=dict(color='black')), row =1, col = 1)

        fig.add_trace(go.Scatter(x = [],y = [], showlegend = True,
                                    name = '',
                                    mode = "lines",
                                    line=dict(color='black')), row = 2, col = 1)

        fig.add_trace(go.Scatter(x = [], y = [], showlegend = False, mode = 'markers', 
                                 marker = dict(color ='red', size = 6)), row = 2, col = 1)  

    fig.update_xaxes(gridcolor='light gray',gridwidth = 0.2,showgrid = True,title_font_size = 16, mirror = True)
    fig.update_yaxes(gridcolor='light gray',gridwidth = 0.2,showgrid = True,title_font_size = 16, mirror = True)
    fig.update_layout(
                        legend = dict(title_text = "Samples",
                                    orientation = 'v',
                                    yanchor="bottom",
                                    y = 1.01,
                                    xanchor = "center",
                                    x = 0.5),
                        template = 'simple_white',
                        height = 500,
                        width = 700,
                        xaxis2 = dict(title_text = "Temperature (℃)"),
                        yaxis = dict(title_text = "350nm/330nm", side = "right"),
                        yaxis2 = dict(title_text = "d/dT(350nm/330nm)"),
                        paper_bgcolor=bgcol
                    )
    return fig

def plot_all(names, data_df, Tm_dict):
    import matplotlib.pyplot as plt
    from plotly.subplots import make_subplots

    # Create subplots based on number of plotted experiments (names - skipped)
    
    names_lower = [n.lower() for n in names]
    num_plotted = len(names) - names_lower.count('skip')
    
    if num_plotted > 3:
        if num_plotted % 3 == 0:
            tot_rows = num_plotted//3
        else:
            tot_rows = num_plotted//3 + 1
        fig,ax = plt.subplots(nrows = tot_rows, 
                              ncols = 3, 
                              figsize = (20,5*tot_rows), 
                              dpi=300, 
                              sharey=True, 
                              sharex=True,
                              squeeze=False)
    else:
        tot_rows = 1
        fig,ax = plt.subplots(nrows = tot_rows, 
                              ncols = num_plotted,
                              figsize = (20*(num_plotted/3),5*tot_rows), 
                              dpi=300, 
                              sharey=True, 
                              sharex=True,
                              squeeze=False)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Delete axes for each subplot so only the top and bottom inset axes are shown
    if tot_rows > 1:
        for row in range(0,tot_rows):
            for col in range(0,3):
                ax[row,col].axis('off')
    
    if tot_rows == 1:
        for col in range(0,num_plotted):
            ax[0,col].axis('off')
    
    count = 0
    for name in names:
        if name.casefold() != 'skip'.casefold():
            # Identify row and column for each plot and extract the appropriate data
            row = (count // 3)
            col = count % 3
            x = data_df[name,'x']
            y = data_df[name,'y']
            dy = data_df[name,'dy']
            peaks = Tm_dict[name + '_peaks']
            Tm = Tm_dict[name + '_Tm']

            # Create top and bottom inset axes for plotting 
            # 350/330 ratio in the top half and 
            # d/dT(350/330 ratio) in the bottom half
            axtop = ax[row,col].inset_axes([0, 0.5, 1.0, 0.5])
            axbot = ax[row,col].inset_axes([0, 0, 1.0, 0.5])


            # Use a dashed line to separate top and bottom plots
            axtop.spines['bottom'].set_visible(False)
            axbot.spines['top'].set_linestyle((10, (1, 5)))
            
            # Remove axis labels and use the experiment name as the title
            axbot.set_ylabel('')
            axtop.set_ylabel('')
            axtop.set_title(name, fontsize=16)
            
            # Show the top panel y-axis labels on the right
            axtop.tick_params(bottom=False,labelbottom=False,right=True,labelright=True,left=False,labelleft=False, labelsize=14)
            axbot.tick_params(bottom=True,labelbottom=True,right=False,labelright=False,left=True,labelleft=True, labelsize=14)
            # Plot the top panel data
            axtop.plot(x,y) 

            
            # Find peaks and plot an asterisk to identify each
            axbot.plot(x,dy)
            axbot.plot(x[peaks],(1.1*dy[peaks]),"*")

            # Find Tm for largest peak and annotate the graph with it
            if peaks.size != 0:
                Tm_y = dy[peaks].max()
                axbot.annotate('Tm = {:3.2f}'.format(Tm), xy=(Tm,Tm_y), xytext=(Tm*1.15, Tm_y * 0.85), fontsize=14)

            # Create y- and x-axes labels for the entire figure using dummy axes 
            # Note that this convoluted method seems necessary to create right and left y-axis labels
            
            # dummy axes 1 for left ylabel
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_xticks([])
            ax1.set_yticks([])
            [ax1.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
            ax1.patch.set_visible(False)
            ax1.set_xlabel('Temperature (℃)', labelpad=40, fontsize = 20)
            ax1.set_ylabel('d/dT(350nm/330nm)', labelpad=60, fontsize = 20, rotation = 90)

            # dummy axes 2 for right ylabel
            ax2 = fig.add_subplot(1, 1, 1)
            ax2.set_xticks([])
            ax2.set_yticks([])
            [ax2.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
            ax2.patch.set_visible(False)
            ax2.yaxis.set_label_position('right')
            ax2.set_ylabel('350nm/330nm', labelpad=60, fontsize = 20,fontweight = 'normal', rotation = 270)
            count+=1
    return plt, fig
