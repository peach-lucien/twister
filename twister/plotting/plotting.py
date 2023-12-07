"""plotting functions."""
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm

import itertools

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.cluster.hierarchy import dendrogram, linkage

from twister.plotting.utils import check_folder, circular_hist

matplotlib.use("Agg")
L = logging.getLogger(__name__)


def _save_to_pdf(pdf, figs=None):
    """Save a list of figures to a pdf."""
    if figs is not None:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close("all")
    else:
        pdf.savefig(bbox_inches="tight")
        plt.close()


def plot(all_results, patient_collection, plotting_args):
    """ Plot all patients in patient collection """
    
    # extract plotting arguments
    folder = plotting_args['plotting_folder']
    ext = plotting_args['ext']
    
    for patient in tqdm(patient_collection):
        results = all_results[patient.patient_id]
        
        # update folder name
        patient_folder = os.path.join(folder,patient.patient_id + '/')
        
        # check folder exists or make
        check_folder(patient_folder)
        
        # plot analysis for individual patient
        plot_analysis(results, patient_folder, patient, ext=ext)

        

def plot_analysis(
    analysis_results,
    folder,
    patient,
    ext=".svg",
):
    """Plot summary of twister analysis."""
    
    with PdfPages(os.path.join(folder, "analysis_report_{}.pdf".format(patient.patient_id))) as pdf:
        
         L.info("Plot movement correlation heatmap")
         try_except_pass(_plot_movement_correlation_heatmap, analysis_results, ext)
         try_except_pass(_save_to_pdf, pdf)

         L.info("Plot mean correlation per movement")
         try_except_pass(_plot_distribution_movement_correlations, analysis_results, ext)
         try_except_pass(_save_to_pdf, pdf)
         
         L.info("Plot symmetry")
         try_except_pass(_plot_symmetry, analysis_results, ext)
         try_except_pass(_save_to_pdf, pdf)
 
         L.info("Plot time spent in each movement")
         try_except_pass(_plot_time_per_movement, analysis_results, ext)
         try_except_pass(_save_to_pdf, pdf)
        
         L.info("Plot movement transitions")
         try_except_pass(_plot_time_per_movement, analysis_results, ext)
         try_except_pass(_save_to_pdf, pdf)

         L.info("Plot movement transition network")
         try_except_pass(_plot_movement_transition_network, analysis_results, ext)
         try_except_pass(_save_to_pdf, pdf)

         L.info("Plot incoming and outgoing movement transitions")
         try_except_pass(_plot_distribution_movement_transitions, analysis_results, ext)
         try_except_pass(_save_to_pdf, pdf)#

         #L.info("Plot scores")
         #_plot_movement_score(analysis_results, ext=ext)
         #_save_to_pdf(pdf)             

         L.info("Plot angles distributions by movement axes")
         try_except_pass(_plot_movement_angles, analysis_results, ext)
         try_except_pass(_save_to_pdf, pdf)

         L.info("Plot angles distributions by movement")
         try_except_pass(_plot_angle_movements, analysis_results, ext)
         try_except_pass(_save_to_pdf, pdf)

         L.info("Plot marker correlationns")
         try_except_pass(_plot_marker_correlation_heatmap, analysis_results, ext)
         try_except_pass(_save_to_pdf, pdf)     
  
         L.info("Plot mean correlation per marker")
         try_except_pass(_plot_distribution_marker_correlations, analysis_results, ext)
         try_except_pass(_save_to_pdf, pdf)
         
         L.info("Plot structural features over time")
         try_except_pass(_plot_structural_features, analysis_results, ext)
         try_except_pass(_save_to_pdf, pdf)

    
def _plot_structural_features(results, ext=".png"):
    """ distribution of marker correlations as boxplot """
    
    # get structural_features
    structural_features = results['MarkerFeatures']['structural_features']
    
    # creating subplot for each structural feature
    fig, axes = plt.subplots(structural_features.shape[1]-1, 1, figsize=(20,30))
    
    # iterate through colour palette
    palette = itertools.cycle(sns.color_palette())
    
    # loop over each structural feature and plot
    for i, col in enumerate(structural_features.drop('time',axis=1).columns):
        sns.lineplot(data=structural_features, x='time', y=col, ax=axes[i], color=next(palette), linewidth = 2) 
        axes[i].set_title(col, rotation=0)
        axes[i].set_ylabel('', rotation=0)
        axes[i].set_xlabel('time (seconds)')
        
    # create sufficient spacing between plots
    fig.tight_layout()

    plt.suptitle('Structural features')  
    
    
    
def _plot_distribution_marker_correlations(results, ext=".png"):
    """ distribution of marker correlations as boxplot """
    
    # get correlation matrix
    correlation_matrix = results['MarkerCorrelations']['correlation_matrix']
    
    # set diagonal to nan
    correlation_matrix.values[[np.arange(correlation_matrix.shape[0])]*2] = np.nan    
    
    plt.figure(figsize=(20,6))    
    sns.barplot(x="bodyparts", y="value", data=correlation_matrix.melt(),
                     palette="Blues_d")

    plt.xlabel('Marker')
    plt.ylabel('Pearson Correlation')
    plt.title('Marker correlation distributions')
  
def _plot_marker_correlation_heatmap(results, ext=".png"):
    """ plot marker correlation heatmap """ 
    
    # extract correlation matrix
    correlation_matrix = results['MarkerCorrelations']['correlation_matrix']
    
    # mask the upper triangular (since its symmetric)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    #Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap.set_bad("grey",alpha=0.5)         
    
    #Draw the heatmap with the mask and correct aspect ratio
    plt.figure(figsize=(6,6))
    cbar_kws = {'label': 'Pearson correlation', "shrink": .5}
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0, #vmax=0.3, vmin=-0.001, 
                    square=True, linewidths=.5, cbar_kws=cbar_kws)
    
    plt.xlabel('Marker')
    plt.ylabel('Marker')
    plt.title('Marker correlations')     
        
    
def _plot_movement_angles(results, ext=".png"):
    """ plotting angle distributions by movement axes """
    
    movements = results['Transitions']['transition_matrix'].columns.tolist()
    movement_axes = ['angle_anteroretrocollis', 'angle_laterocollis', 'angle_torticollis']

    fig, ax = plt.subplots(1,3, figsize=(20,30), subplot_kw=dict(projection='polar'))
    cmap = plt.get_cmap('tab10')    
    
    angle_axes = movement_axes[1]
    for k, angle_axes in enumerate(movement_axes):
        plotted_movements = []

        for i, movement in enumerate(movements):        
            dist = results['Angles'][angle_axes][movement][angle_axes+'_distribution']
            if dist is not None:
                dist = np.rad2deg(dist.dropna())            
                circular_hist(ax[k], np.deg2rad(dist), offset=np.pi/2, color=cmap.colors[i] ,)
            
                plotted_movements.append(movement)
            
        ax[k].set_title(angle_axes)
        ax[k].legend(plotted_movements)


def _plot_angle_movements(results, ext=".png"):
    """ plotting angle distributions by movement """
    
    movements = results['Transitions']['transition_matrix'].columns.tolist()
    movement_axes = ['angle_anteroretrocollis', 'angle_laterocollis', 'angle_torticollis']

    fig, ax = plt.subplots(7,1, figsize=(60,35), subplot_kw=dict(projection='polar'))
    cmap = plt.get_cmap('tab10')    
    
    angle_axes = movement_axes[1]
    for k, movement in enumerate(movements):

        for i, angle_axes in enumerate(movement_axes):        
            dist = results['Angles'][angle_axes][movement][angle_axes+'_distribution']
            if dist is not None:
                dist = np.rad2deg(dist.dropna())            
                circular_hist(ax[k], np.deg2rad(dist), offset=np.pi/2, color=cmap.colors[i] ,)
            
            
        ax[k].set_title(movement)
        ax[k].legend(movement_axes)
        
    plt.tight_layout()

def _plot_movement_score(results, ext=".png"):
    """ plotting score distributions """

    movements = results['Transitions']['transition_matrix'].columns.drop('face_forward')
    #score_distribution = results['Scores']['feature_vector']    
    
    fig, axes = plt.subplots(6, 1, figsize=(6, 8), sharex=True)
    
    movement_results = pd.DataFrame(columns = movements)
    for i, movement in enumerate(movements):
        
        movement_results.loc[0,movement] = results['Scores'][movement]['median']
        
        
        # sns.barplot(x=list(score_distribution.columns), 
        #             y=score_distribution.loc[movement,:].values,
        #             ax=axes[i], palette="Spectral", order=list(score_distribution.columns))  
        
        # axes[i].set_ylim([0,1])
        # axes[i].set_ylabel(movement)
        # #axes[i].bar_label(axes[i].containers[0],label_type='center')
        
        # # Adjust width    
        # for patch in axes[i].patches:
        #     current_width = patch.get_width()
        #     patch.set_width(1)
        #     patch.set_y(patch.get_y() + current_width - 1)
    
    movement_results = movement_results.dropna(axis=1)
    
    plt.figure(figsize=(10,6))
    sns.barplot(data=movement_results)
    plt.ylim([0,5])
            
    #plt.xlabel('Score (worst to best, 4-0)') 
    plt.suptitle('Movement Score')    
    plt.ylabel('Score')
    
def _plot_time_per_movement(results, ext=".png"):
    """ plotting time spent in each movement in seconds """ 
    
    # get movement times
    movement_t = results['Symmetry']['sum_movements']

    plt.figure(figsize=(10,6))

    # define palette
    palette = sns.color_palette("YlOrBr", movement_t.shape[1]) 
    
    # ranking colours by largest on y axis
    rank = movement_t.values.argsort().argsort().reshape(-1)

    ax = sns.barplot(x="variable", y="value", data=movement_t.melt(),
                     palette=[palette[r] for r in rank])   
    
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
    
    plt.xlabel('Movement')
    plt.ylabel('Time (seconds)')
    plt.title('Time spent in each movement')
    
    
def _plot_distribution_movement_transitions(results, ext=".png"):
    """ distribution of movement correlations as boxplot """
    
    # get correlation matrix
    transition_matrix = results['Transitions']['transition_matrix']
    
    # set diagonal to nan
    transition_matrix.values[[np.arange(transition_matrix.shape[0])]*2] = np.nan  
    
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15,4)) 
    
    # plot incoming probability 
    sns.barplot(ax=axes[0], x="variable", y="value", data=transition_matrix.melt(),
                     palette="Blues_d")

    axes[0].set_xlabel('Movement')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Movement incoming transition probability')
    
    # outgoing probability
    sns.barplot(ax=axes[1], x="variable", y="value", data=transition_matrix.T.melt(),
                     palette="Blues_d")
    
    axes[1].set_xlabel('Movement')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Movement outgoing transition probability')
    
    
    
    
    
def _plot_movement_transitions(results, ext=".png"):
    """ plot transition matrix heatmap """ 

    # extract movement transitions
    transition_matrix = results['Transitions']['transition_matrix']
        
    # mask the upper triangular (since its symmetric)
    mask = np.triu(np.ones_like(transition_matrix, dtype=bool))

    #Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap.set_bad("grey",alpha=0.5)         
    
    #Draw the heatmap with the mask and correct aspect ratio
    plt.figure(figsize=(6,6))
    cbar_kws = {'label': 'Transition probability', "shrink": .5}
    sns.heatmap(transition_matrix, mask=mask, cmap=cmap, center=0, #vmax=0.3, vmin=-0.001, 
                    square=True, linewidths=.5, cbar_kws=cbar_kws)
    
    plt.xlabel('Movement')
    plt.ylabel('Movement')
    plt.title('Movement transition matrix')
    

    
def _plot_movement_transition_network(results, ext=".png"):
    """ plot transition matrix heatmap """ 

    # extract movement transitions
    transition_matrix = results['Transitions']['n_transitions']
       
    T_ = transition_matrix.copy()
    np.fill_diagonal(T_.values,0)
    T = nx.DiGraph(T_)
    
    pos = nx.circular_layout(T)
    edges = T.edges()
    colors = [T[u][v]['weight'] for u,v in edges]
    weights = [T[u][v]['weight'] for u,v in edges]
    node_size = np.diagonal(transition_matrix)    
    

    plt.figure(figsize=(8,6))

    # Draw nodes and edges
    nx.draw_networkx_nodes(T, pos, node_size=node_size)
    nx.draw_networkx_labels(T, pos)
    if weights:
        nx.draw_networkx_edges(
            T, pos,
            connectionstyle="arc3,rad=0.1",
            width=10*(weights/np.max(weights)),
            edge_color=colors,
        )
    
    plt.title('Movement transition network')    
    
def _plot_symmetry(results, ext=".png"):
    
    # get movement symmetries
    movement_symmetry = results['Symmetry'].copy()
    
    # remove sum of movements
    movement_symmetry.pop('sum_movements', None)
    movement_symmetry.pop('feature_vector', None)

    #construct dataframe
    data = pd.DataFrame(movement_symmetry,index=[0]).melt()
   
    # defining the right and left ticks for plotting
    right_ticks = ['head_chest','rot_right','tilt_right'] 
    left_ticks = ['head_down','rot_left','tilt_left'] 
   
    plt.figure()
    ax = sns.barplot(x="value", y="variable", data=data)
    
    twin_ax = ax.twinx()
    twin_ax.set_yticks([0,1,2],right_ticks)
    twin_ax.set_ylim(ax.get_ylim())
    ax.set_yticks([0,1,2],left_ticks)
    
    plt.xlim([-1.1, 1.1])    
    plt.title('Movement symmetry')
    plt.xlabel('Symmetry')
    plt.ylabel('Movement')

def _plot_distribution_movement_correlations(results, ext=".png"):
    """ distribution of movement correlations as boxplot """
    
    # get correlation matrix
    correlation_matrix = results['Correlations']['correlation_matrix']
    
    # set diagonal to nan
    correlation_matrix.values[[np.arange(correlation_matrix.shape[0])]*2] = np.nan    
    
    plt.figure(figsize=(10,6))    
    sns.barplot(x="variable", y="value", data=correlation_matrix.melt(),
                     palette="Blues_d")

    plt.xlabel('Movement')
    plt.ylabel('Pearson Correlation')
    plt.title('Movement correlation distributions')
    
    
def _plot_movement_correlation_heatmap(results, ext=".png"):
    """ plot correlation heatmap """ 
    
    # extract correlation matrix
    correlation_matrix = results['Correlations']['correlation_matrix']
    
    # mask the upper triangular (since its symmetric)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    #Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap.set_bad("grey",alpha=0.5)         
    
    #Draw the heatmap with the mask and correct aspect ratio
    plt.figure(figsize=(6,6))
    cbar_kws = {'label': 'Pearson correlation', "shrink": .5}
    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0, #vmax=0.3, vmin=-0.001, 
                    square=True, linewidths=.5, cbar_kws=cbar_kws)
    
    plt.xlabel('Movement')
    plt.ylabel('Movement')
    plt.title('Movement correlations')
    
    
def try_except_pass(func, *args, **kwargs):
    """
    Wrapper function to execute a function within a try-except block.

    Parameters:
    - func: The function to be executed.
    - *args: Positional arguments to pass to the function.
    - **kwargs: Keyword arguments to pass to the function.

    Returns:
    - Result of the function call if successful, None otherwise.
    """
    try:
        return func(*args, **kwargs)
    except:
        #print(f"An error occurred in function {func.__name__}: {e}")
        pass  # or handle the exception as needed


