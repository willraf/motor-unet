"""
Modular visualization framework for signal processing visualization.

This framework provides reusable, composable visualization components
organized on a grid-based layout system. Components can be easily combined
to create complex visualizations with minimal code duplication.

Usage:
    fig = ModularFigure(n_rows=3, n_cols=2, figsize=(14, 10))
    fig.add_component(SignalPlot(signals, channels=['Ch1', 'Ch2']), 
                      row_slice=slice(0, 2), col=0)
    fig.add_component(SpikeTrain(spikes), row_slice=slice(2, 3), col=0)
    fig.save_or_show("output.pdf")
"""

import logging
from abc import ABC, abstractmethod
from typing import Union, Optional, List, Tuple, Type, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
import torch
import math

from pathlib import Path


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def convert_to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert torch.Tensor to numpy array if needed."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def remove_spines(ax: plt.Axes) -> None:
    """Remove all spines from axis."""
    for spine in ax.spines.values():
        spine.set_visible(False)


def format_signal_axis(ax: plt.Axes, show_ticks: bool = False) -> None:
    """Apply standard formatting to signal axes."""
    remove_spines(ax)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])


# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================


class VisualizationComponent(ABC):
    """Base class for all visualization components."""
    
    @abstractmethod
    def plot(self, fig: plt.Figure, gs: gridspec.GridSpec) -> None:
        """Plot component on given figure and gridspec.
        
        Args:
            fig: Matplotlib figure to draw on.
            gs: Gridspec defining the area to draw in.
        """
        pass


class ComponentContainer(VisualizationComponent):

    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.components = []
        self.fig = None
        self.gs = None


    def add_component(self,
                     component: Union["VisualizationComponent", "ComponentContainer"],
                     row_slice: Union[int, slice],
                     col_slice: Union[int, slice]) -> None:
        """Add visualization component to grid.
        
        Args:
            component: Component to add.
            row_slice: Row index or slice (e.g., 0 or slice(0, 2)).
            col_slice: Column index or slice.
        """
        # Convert single indices to slices
        if isinstance(row_slice, int):
            row_slice = slice(row_slice, row_slice + 1)
        if isinstance(col_slice, int):
            col_slice = slice(col_slice, col_slice + 1)

        self.components.append((component, row_slice, col_slice))


    def build_grid(self, fig: plt.Figure, parent_spec: gridspec.GridSpec = None):
        """Initialize figure and gridspec.
        
        Args:
            fig: Matplotlib figure to draw on.
            parent_spec: Optional parent gridspec to nest within.
        """
        self.fig = fig
        if parent_spec is None:
            self.gs = fig.add_gridspec(self.n_rows, self.n_cols)
        else:
            self.gs = parent_spec.subgridspec(self.n_rows, self.n_cols)


    def plot(self, fig: plt.Figure, parent_gs: gridspec.GridSpec = None):
        """Draw this container and its children.
        
        Args:
            fig: Matplotlib figure to draw on.
            parent_gs: Optional parent gridspec to nest within.
        """
        self.build_grid(fig, parent_gs)

        for component, row_slice, col_slice in self.components:
            component.plot(self.fig, self.gs[row_slice, col_slice])


class ModularFigure(ComponentContainer):
    """Grid-based figure manager for composing visualizations.
    
    Args:
        n_rows: Number of grid rows.
        n_cols: Number of grid columns.
        figsize: Figure size (width, height).
        **kwargs: Additional arguments passed to plt.figure().
    """
    
    def __init__(self, 
                 n_rows: int,
                 n_cols: int,
                 figsize: Tuple[float, float] = (14, 10),
                 **kwargs):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.fig = plt.figure(figsize=figsize, **kwargs)
        self.gs = gridspec.GridSpec(n_rows, n_cols, figure=self.fig)
        self.components = []
    
    def set_title(self, title: str, **kwargs) -> None:
        """Add figure title."""
        self.fig.suptitle(title, **kwargs)
    
    def tight_layout(self) -> None:
        """Apply tight layout."""
        plt.tight_layout()
    
    def save_or_show(self, fpath: Optional[str] = None) -> None:
        """Save figure or display.
        
        Args:
            fpath: Path to save figure. If None, display instead.
        """
        for component, row_slice, col_slice in self.components:
            component.plot(self.fig, self.gs[row_slice, col_slice])

        if fpath:
            import os
            os.makedirs(os.path.dirname(fpath) or '.', exist_ok=True)
            self.fig.savefig(fpath, transparent=True, bbox_inches='tight')
            logging.info(f"Saved plot to {fpath}")
        else:
            plt.show()
        plt.close(self.fig)


    def plot(self, fpath: Optional[str] = None) -> None:
        """Draw all components in the figure."""
        for component, row_slice, col_slice in self.components:
            component.plot(self.fig, self.gs[row_slice, col_slice])

        self.save_or_show(fpath)

# ============================================================================
# Custom Visualization Components
# ============================================================================


class SignalPlot(VisualizationComponent):
    """Plot multi-channel signals with optional vertical offset.
    
    Args:
        signals: Array of shape (num_channels, time_steps) or (time_steps,).
        channel_labels: Optional labels for each channel.
        offset: Vertical offset between channels.
        colors: Optional color per channel or single color for all.
        linewidth: Line width for signal plots.
    """
    
    def __init__(self, 
                 signals: Union[np.ndarray, torch.Tensor],
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 channel_labels: Optional[List[str]] = None,
                 offset: float = 1.0,
                 colors: Optional[Union[str, List[str]]] = None,
                 linewidth: float = 1.0):
        
        self.signals = convert_to_numpy(signals)
        if self.signals.ndim == 1:
            self.signals = self.signals[np.newaxis, :]
        
        self.num_channels = self.signals.shape[0]
        self.channel_labels = channel_labels or [f'{i+1}' for i in range(self.num_channels)]
        self.offset = offset
        if colors is None:
            self.colors = None
        else:
            self.colors = [colors] * self.num_channels if isinstance(colors, str) else colors
        self.linewidth = linewidth

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self, fig: plt.Figure, gs: gridspec.GridSpec) -> None:
        """Plot signals with offset."""
        time = np.arange(self.signals.shape[1])
        ax = fig.add_subplot(gs)
        for ch in range(self.num_channels):
            if self.colors is None:
                color = None
            else:
                color = self.colors[ch % len(self.colors)]

            ax.plot(time, 
                    self.signals[ch] + ch * self.offset, 
                    label=self.channel_labels[ch], 
                    color=color, 
                    linewidth=self.linewidth)

        if self.title:
            ax.set_title(self.title)
        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)

        ax.set_yticks(np.arange(self.num_channels) * self.offset)
        ax.set_yticklabels(self.channel_labels)
        # format_signal_axis(ax)


class RepeatedPlot(ComponentContainer):
    """Plot multiple instances of the same component in a grid.
    
    Args:
        ComponentClass: Class of component to repeat.
        component_data: List of data inputs for each component instance.
        kwargs: Additional arguments passed to component constructor.
        n_rows: Number of rows in grid.
        n_cols: Number of columns in grid.
    """
    
    def __init__(self, 
                 ComponentClass: Type[VisualizationComponent],
                 component_data: List[Any],
                 n_rows: int = None,
                 n_cols: int = None,
                 labels: Optional[List[str]] = None,
                 label_format: Optional[str] = None,
                 label_mode: str = 'title',
                 **kwargs):

        self.num_components = len(component_data)

        if n_rows is None and n_cols is None:
            n_rows, n_cols = 1, self.num_components
        elif n_rows is None:
            n_rows = math.ceil(self.num_components / n_cols)
        elif n_cols is None:
            n_cols = math.ceil(self.num_components / n_rows)

        if n_rows * n_cols < self.num_components:
            raise ValueError("Grid size too small for number of components")

        super().__init__(n_rows=n_rows, n_cols=n_cols)

        # Validate label_mode
        if label_mode not in ('title', 'xlabel', 'ylabel'):
            raise ValueError("label_mode must be 'title', 'xlabel', or 'ylabel'")


        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                data = component_data[idx]

                # Create component with base kwargs
                component_kwargs = kwargs.copy()

                if labels is not None:
                    label = labels[idx]
                elif label_format:
                    label = label_format.format(i=idx+1)
                else:
                    label = None

                # Apply auto-labeling if requested
                if label is not None:
                    if label_mode == 'title':
                        component_kwargs['title'] = label
                    elif label_mode == 'xlabel':
                        component_kwargs['xlabel'] = label
                    elif label_mode == 'ylabel':
                        component_kwargs['ylabel'] = label
                
                component = ComponentClass(data, **component_kwargs)
                self.add_component(component, row_slice=r, col_slice=c)


class SignalGrid(ComponentContainer):

    def __init__(self, 
                 signals: Union[np.ndarray, torch.Tensor],
                 channel_labels: Optional[List[str]] = None,
                 offset: float = 1.0,
                 colors: Optional[Union[str, List[str]]] = None,
                 linewidth: float = 1.0):
        
        self.signals = convert_to_numpy(signals)
        if self.signals.ndim == 2:
            self.signals = self.signals[np.newaxis, :, :]
        
        self.num_signals = self.signals.shape[0]
        self.num_channels = self.signals.shape[1]
        self.channel_labels = channel_labels or [f'{i+1}' for i in range(self.num_channels)]
        self.offset = offset
        self.colors = [colors] * self.num_channels if isinstance(colors, str) else colors
        self.linewidth = linewidth

        super().__init__(n_rows=self.num_signals, n_cols=1)

        for signal_idx in range(self.num_signals):
            self.add_component(
                SignalPlot(self.signals[signal_idx], 
                           channel_labels=self.channel_labels, 
                           offset=self.offset, 
                           colors=self.colors, 
                           linewidth=self.linewidth),
                row_slice=signal_idx, col_slice=0)


class SignalOverlay(VisualizationComponent):
    """Plot multiple signals on same axis with transparency/color differentiation.
    
    Args:
        signals_dict: Dict mapping signal name to (array, color, alpha, linewidth).
        offset: Vertical offset between channels.
    """
    
    def __init__(self, 
                 signals_dict: dict,
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None,
                 channel_labels: Optional[List[str]] = None,
                 offset: float = 1.0):
        self.signals_dict = {name: (convert_to_numpy(sig), color, alpha, lw)
                            for name, (sig, color, alpha, lw) in signals_dict.items()}
        self.signals_dict = {name: (sig if sig.ndim == 2 else sig[np.newaxis, :], color, alpha, lw) 
                             for name, (sig, color, alpha, lw) in self.signals_dict.items()}
        
        self.num_channels = max([sig.shape[0] for sig, _, _, _ in self.signals_dict.values()])
        # self.num_channels = next(iter(self.signals_dict.values()))[0].shape[0]
        self.channel_labels = channel_labels or [f'Ch {i+1}' for i in range(self.num_channels)]

        self.offset = offset
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self, fig: plt.Figure, gs: gridspec.GridSpec) -> None:
        """Plot overlaid signals."""
        ax = fig.add_subplot(gs)
        time = np.arange(next(iter(self.signals_dict.values()))[0].shape[-1])
        for name, (signal, color, alpha, lw) in self.signals_dict.items():

            for ch in range(signal.shape[0]):
                ax.plot(time, 
                        signal[ch] + ch * self.offset,
                        label=self.channel_labels[ch], 
                        color=color, alpha=alpha, linewidth=lw)
        # format_signal_axis(ax)


class SpikeTrain(VisualizationComponent):
    """Plot binary spike trains.
    
    Args:
        spike_trains: Array of shape (num_sources, time_steps).
        source_labels: Optional labels for each source.
        colors: Line colors.
        linewidth: Line width.
    """
    
    def __init__(self,
                 spike_trains: Union[np.ndarray, torch.Tensor],
                 source_labels: Optional[List[str]] = None,
                 colors: str = 'blue',
                 linewidth: float = 0.8):

        self.spike_trains = convert_to_numpy(spike_trains)
        if self.spike_trains.ndim == 1:
            self.spike_trains = self.spike_trains[np.newaxis, :]

        self.num_sources = self.spike_trains.shape[0]
        self.source_labels = source_labels or [f'Source {i+1}' for i in range(self.num_sources)]
        self.colors = colors
        self.linewidth = linewidth

    def plot(self, fig: plt.Figure, gs: gridspec.GridSpec) -> None:
        """Plot spike trains as vertical lines."""
        ax = fig.add_subplot(gs)
        for source in range(self.num_sources):
            spike_times = np.where(self.spike_trains[source] > 0)[0]
            ax.vlines(spike_times, source - 0.5, source + 0.5, color=self.colors, linewidth=self.linewidth)
        
        ax.set_yticks(range(self.num_sources))
        ax.set_yticklabels(self.source_labels)   
        ax.set_ylim(-0.5, self.num_sources - 0.5)
        ax.set_xlim(0, self.spike_trains.shape[1])
        # format_signal_axis(ax)


class HeatmapPlot(VisualizationComponent):
    """Plot 2D heatmap.
    
    Args:
        data: 2D array.
        cmap: Colormap name.
        vmin/vmax: Value range for normalization.
    """
    
    def __init__(self,
                 data: Union[np.ndarray, torch.Tensor],
                 cmap: str = 'viridis',
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None):
        self.data = convert_to_numpy(data)
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
    
    def plot(self, fig: plt.Figure, gs: gridspec.GridSpec) -> None:
        """Plot heatmap."""
        ax = fig.add_subplot(gs)
        im = ax.imshow(self.data, aspect='auto', cmap=self.cmap, 
                      vmin=self.vmin, vmax=self.vmax)
        format_signal_axis(ax)


class SaliencyOverlay(VisualizationComponent):
    """Overlay saliency map on signal with colored line segments.
    
    Args:
        signal: Signal array of shape (num_channels, time_steps).
        saliency: Saliency array of shape (num_channels, time_steps).
        offset: Vertical offset between channels.
        cmap: Colormap for saliency.
    """
    
    def __init__(self,
                 signal: Union[np.ndarray, torch.Tensor],
                 saliency: Union[np.ndarray, torch.Tensor],
                 offset: float = 5.0,
                 cmap: str = 'Reds'):
        self.signal = convert_to_numpy(signal)
        self.saliency = convert_to_numpy(saliency)
        self.saliency = self.saliency / (self.saliency.max() + 1e-6)
        self.offset = offset
        self.cmap = cmap
        if self.signal.ndim == 1:
            self.signal = self.signal[np.newaxis, :]
        if self.saliency.ndim == 1:
            self.saliency = self.saliency[np.newaxis, :]
    
    def plot(self, fig: plt.Figure, gs: gridspec.GridSpec) -> None:
        """Plot signal with saliency overlay."""
        from matplotlib.collections import LineCollection
        ax = fig.add_subplot(gs)
        
        num_channels, time_steps = self.signal.shape
        cmap = plt.get_cmap(self.cmap)
        
        for ch in range(num_channels):
            offset_ch = ch * self.offset
            sig = self.signal[ch]
            sal = self.saliency[ch] / (self.saliency[ch].max() + 1e-6)
            
            ax.plot(sig + offset_ch, color='black', linewidth=1.0, alpha=0.3)
            
            # Create colored line segments based on saliency
            points = np.array([np.arange(time_steps), sig + offset_ch]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = cmap(sal[:-1])
            
            lc = LineCollection(segments, colors=colors, linewidths=2, alpha=0.7)
            ax.add_collection(lc)
        
        format_signal_axis(ax)


# ============================================================================
# EXAMPLE WRAPPER FUNCTIONS (Optional convenience functions)
# ============================================================================

def plot_emg(emg: Union[np.ndarray, torch.Tensor], 
             fpath: Optional[str] = None, 
             offset: Optional[float] = 1.0) -> None:
    """Quick plot of EMG signal."""
    fig = ModularFigure(1, 1, figsize=(15, 4))
    fig.add_component(SignalPlot(emg, offset=offset), 0, 0)
    fig.save_or_show(fpath)


# def plot_emg_muaps_prediction(emg: Union[np.ndarray, torch.Tensor], 
#                               muaps_true: Union[np.ndarray, torch.Tensor], 
#                               muaps_pred: Union[np.ndarray, torch.Tensor],
#                               offset: float = 1.0, 
#                               fpath: Optional[str] = None) -> None:
#     """Plot EMG signal along with true and predicted MUAPs."""

#     num_sources, num_channels, time_steps = muaps_true.shape
#     fig_width = 2.5 * (num_sources + 1)
#     fig = ModularFigure(2, num_sources + 1, figsize=(fig_width, 10))

#     fig.add_component(SignalPlot(emg, title="Raw EMG", offset=offset), slice(None), 0)
    
#     pred_plot = RepeatedPlot(SignalPlot, component_data=muaps_pred, offset=offset, label_format="Predicted MUAP {i}", label_mode='ylabel')
#     fig.add_component(pred_plot, 0, slice(1, None))

#     gt_plot = RepeatedPlot(SignalPlot, component_data=muaps_true, offset=offset, label_format="GT MUAP {i}", label_mode='ylabel')
#     fig.add_component(gt_plot, 1, slice(1, None))



#     fig.save_or_show(fpath)


# Combined Visualizations
def plot_emg_muaps_prediction(emg: Union[np.ndarray, torch.Tensor],
                            muaps_true: Union[np.ndarray, torch.Tensor],
                            muaps_pred: Union[np.ndarray, torch.Tensor],
                            fpath: Optional[str] = None) -> None:
    """Plot EMG signal alongside true and predicted MUAPs.
    
    Args:
        emg: EMG signal of shape (num_channels, time_steps)
        muaps_true: True MUAPs of shape (num_sources, num_channels, time_steps)
        muaps_pred: Predicted MUAPs of shape (num_sources, num_channels, time_steps)
        fpath: Optional path to save the plot
    """
    emg = _convert_to_numpy(emg)
    muaps_true = _convert_to_numpy(muaps_true)
    muaps_pred = _convert_to_numpy(muaps_pred)
    
    num_sources, num_channels, time_steps = muaps_true.shape

    offset = 1
    
    fig_width = 2.5 * (num_sources + 1)
    fig = plt.figure(figsize=(fig_width, 6))
    gs = gridspec.GridSpec(2, num_sources + 1, figure=fig)
    
    # Plot EMG
    ax_emg = fig.add_subplot(gs[:, 0])
    for ch in range(num_channels):
        ax_emg.plot(emg[ch] + ch * offset, label=f'Ch {ch+1}', linewidth=1.5)
    ax_emg.set_title('Original EMG')
    ax_emg.set_xticks([])
    ax_emg.set_yticks([])
    
    # Plot predicted MUAPs
    for source in range(num_sources):
        ax = fig.add_subplot(gs[0, source + 1])
        for ch in range(num_channels):
            ax.plot(muaps_pred[source, ch] + ch * offset, linewidth=1.5)
        ax.set_ylabel(f'Pred {source+1}', fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Plot true MUAPs
    for source in range(num_sources):
        ax = fig.add_subplot(gs[1, source + 1])
        for ch in range(num_channels):
            ax.plot(muaps_true[source, ch] + ch * offset, linewidth=1.5)
        ax.set_ylabel(f'GT {source+1}', fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    # Remove spines (boxes) from all axes
    for ax in fig.get_axes():
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    plt.tight_layout()
    # plt.suptitle("EMG Signal and MUAP Identification", fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.95)
    
    _save_or_show(fig, fpath)


#     muap_gt = muaps_true[0]
#     muap_pred = muaps_pred[0]

#     plot_signal_embc(muap_gt, f'{fpath}_muap_gt.png')
#     plot_signal_embc(muap_pred, f'{fpath}_muap_pred.png')
#     plot_signal_embc(emg, f'{fpath}_emg.png')

#     residual = emg - muap_pred
#     plot_signal_embc(residual, f'{fpath}_residual.png')

#     gt_residual = emg - muap_gt
#     plot_signal_embc(gt_residual, f'{fpath}_gt_residual.png')


# def plot_signal_embc(signal: Union[np.ndarray, torch.Tensor], fname):

#     C, T = signal.shape
#     fig = plt.figure(figsize=(10, 10))
#     for ch in range(C):
#         plt.plot(signal[ch] + ch * 1.0, label=f'Ch {ch+1}', linewidth=2.9)

#     plt.ylabel('Amplitude + Offset', fontsize=18)
#     plt.xlabel('Time (ms)', fontsize=16)
#     plt.savefig(fname, bbox_inches='tight')
    



# ============================================================================
#                    Non-modular legacy plotting functions
# ============================================================================

def _convert_to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert torch.Tensor to numpy array if needed."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def _save_or_show(fig: plt.Figure, fpath: Optional[str] = None) -> None:
    """Save figure to file if path provided, otherwise show it."""
    if fpath:
        fpath = Path(fpath)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fpath, transparent=True, bbox_inches='tight')
        logging.info(f"Saved plot to {fpath}")
    else:
        plt.show()
    plt.close(fig)


def plot_loss_curves(train_loss: np.ndarray, val_loss: np.ndarray, 
                    epochs: np.ndarray, fpath: Optional[str] = None) -> None:
    """Plot training and validation loss curves.
    
    Args:
        train_loss: Array of training losses
        val_loss: Array of validation losses
        epochs: Array of epoch numbers
        fpath: Optional path to save the plot
    """

    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    line_train, = ax.plot([], [], 'r-', label='Train Loss')
    line_val, = ax.plot([], [], 'b-', label='Validation Loss')
    ax.legend()

    # Interim plot update
    line_train.set_xdata(epochs)
    line_train.set_ydata(train_loss)
    line_val.set_xdata(epochs)
    line_val.set_ydata(val_loss)
    ax.relim()
    ax.autoscale_view()
        
    _save_or_show(fig, fpath)


def plot_nmse(avg_nmse_per_iteration: List[float], fpath: Optional[str] = None):
    """
    Plot NMSE over epochs.
    
    Args:
        normalised_mse: List of NMSE values over epochs.
        fpath: Optional path to save the plot.
    """

    xs = list(range(1, len(avg_nmse_per_iteration) + 1))


    fig, ax = plt.subplots(figsize=(10, 6))


    ax.plot(xs, avg_nmse_per_iteration, marker='s', label='MotorUNet Model NMSE (subject 1)')
    ax.axhline(y=1, color='r', linestyle='--', label='Zero baseline')
    
    ax.set_xlabel("Iteration number (n)", fontsize=14)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_ylabel("Normalised MSE", fontsize=14)
    ax.grid(False)
    ax.legend(fontsize=12)

    _save_or_show(fig, fpath)




# ============================================================================
# TEST AND EXAMPLE FUNCTIONS
# ============================================================================

def generate_test_data():
    """Generate synthetic EMG and related data for testing."""
    np.random.seed(42)
    time_steps = 500
    num_channels = 12
    num_sources = 2
    
    # EMG signal
    emg = np.random.randn(num_channels, time_steps) * 0.5
    
    # MUAPs
    muaps_true = np.random.randn(num_sources, num_channels, 50) * 0.3
    muaps_pred = muaps_true + np.random.randn(num_sources, num_channels, 50) * 0.1
    
    # Spike trains
    spike_trains = np.zeros((num_sources, time_steps))
    for source in range(num_sources):
        spike_indices = np.random.choice(time_steps, size=20, replace=False)
        spike_trains[source, spike_indices] = 1
    
    # Saliency map
    saliency = np.abs(np.random.randn(num_channels, time_steps))
    
    return {
        'emg': emg,
        'muaps_true': muaps_true,
        'muaps_pred': muaps_pred,
        'spike_trains': spike_trains,
        'saliency': saliency,
    }


def test_signal_plot():
    """Test basic signal plotting."""
    data = generate_test_data()
    
    fig = ModularFigure(1, 1, figsize=(12, 4))
    fig.add_component(SignalPlot(data['emg'], offset=2.0), 0, 0)
    fig.set_title("Test: SignalPlot")
    fig.tight_layout()
    fig.save_or_show()


def test_signal_grid():
    """Test signal grid component."""
    data = generate_test_data()
    fig = ModularFigure(1, 1, figsize=(14, 6))
    fig.add_component(SignalGrid(data['muaps_true'], offset=3.0), 0, 0)
    fig.set_title("Test: SignalGrid (MUAPs)")
    fig.tight_layout()
    fig.save_or_show()

def test_repeated_plot():
    """Test repeated plot component."""
    data = generate_test_data()
    muaps = data['muaps_true']
    
    fig = ModularFigure(2, 2, figsize=(10, 8))
    fig.add_component(RepeatedPlot(SignalPlot, 
                                   component_data=[muaps[i] for i in range(muaps.shape[0])],
                                   offset=3.0),
                      row_slice=1, col_slice=0)
    fig.set_title("Test: RepeatedPlot (MUAPs)")
    fig.tight_layout()
    fig.save_or_show()


def test_spike_train():
    """Test spike train plotting."""
    data = generate_test_data()
    
    fig = ModularFigure(1, 1, figsize=(12, 4))
    fig.add_component(SpikeTrain(data['spike_trains']), 0, 0)
    fig.set_title("Test: SpikeTrain")
    fig.tight_layout()
    fig.save_or_show()


def test_signal_overlay():
    """Test overlaying predicted and true signals."""
    data = generate_test_data()
    
    # Average MUAPs across channels to get 1D source signals
    true_source = np.mean(data['muaps_true'], axis=1)
    pred_source = np.mean(data['muaps_pred'], axis=1)
    
    overlay_dict = {
        'Ground Truth': (true_source, 'blue', 1.0, 2.0),
        'Predicted': (pred_source, 'red', 0.7, 1.5),
    }
    
    fig = ModularFigure(1, 1, figsize=(12, 5))
    fig.add_component(SignalOverlay(overlay_dict, offset=3.0), 0, 0)
    fig.set_title("Test: SignalOverlay (Prediction vs Ground Truth)")
    fig.tight_layout()
    fig.save_or_show()


def test_saliency_overlay():
    """Test saliency map overlay."""
    data = generate_test_data()
    
    fig = ModularFigure(1, 1, figsize=(12, 5))
    fig.add_component(SaliencyOverlay(data['emg'], data['saliency'], offset=3.0), 0, 0)
    fig.set_title("Test: SaliencyOverlay")
    fig.tight_layout()
    fig.save_or_show()


def test_complex_layout():
    """Test complex multi-panel layout."""
    data = generate_test_data()
    num_sources = data['muaps_true'].shape[0]
    
    fig = ModularFigure(3, num_sources + 1, figsize=(14, 10))
    
    # EMG on left spanning all rows
    fig.add_component(SignalPlot(data['emg'], offset=2.0),
                     row_slice=slice(0, 3), col_slice=0)
    
    # Predicted MUAPs
    for source in range(num_sources):
        fig.add_component(SignalPlot(data['muaps_pred'][source], offset=3.0),
                         row_slice=0, col_slice=source + 1)
    
    # True MUAPs
    for source in range(num_sources):
        fig.add_component(SignalPlot(data['muaps_true'][source], offset=3.0),
                         row_slice=1, col_slice=source + 1)
    
    # Spike trains
    for source in range(num_sources):
        fig.add_component(SpikeTrain(data['spike_trains'][source:source+1]),
                         row_slice=2, col_slice=source + 1)
    
    fig.set_title("Test: Complex Multi-Panel Layout", fontsize=14)
    fig.tight_layout()
    fig.save_or_show()


if __name__ == '__main__':
    print("Running visualization tests...\n")
    
    print("1. Testing SignalPlot...")
    test_signal_plot()
    
    print("2. Testing SignalGrid...")
    test_signal_grid()
    
    print("3. Testing SpikeTrain...")
    test_spike_train()
    
    print("4. Testing SignalOverlay...")
    test_signal_overlay()
    
    print("5. Testing SaliencyOverlay...")
    test_saliency_overlay()
    
    print("6. Testing Complex Multi-Panel Layout...")
    test_complex_layout()

    print("7. Testing RepeatedPlot...")
    test_repeated_plot()

    print("\nAll tests completed!")

# Spike Train Visualization
def plot_spike_trains(spike_trains: Union[np.ndarray, torch.Tensor], fpath: Optional[str] = None) -> None:
    """Plot spike trains for each source.
    
    Args:
        spike_trains: Binary spike trains of shape (num_sources, time_steps)
        fpath: Optional path to save the plot
    """
    spike_trains = _convert_to_numpy(spike_trains)
    num_sources, time_steps = spike_trains.shape
    
    fig, ax = plt.subplots(figsize=(14, 6))
    time = np.arange(time_steps)
    
    for source in range(num_sources):
        spike_times = np.where(spike_trains[source] == 1)[0]
        ax.vlines(spike_times, source, source + 1, color='blue', linewidth=0.8)
    
    ax.set_title('Spike Trains')
    ax.set_xlabel('Time')
    ax.set_ylabel('Source')
    ax.set_yticks(range(num_sources + 1))
    ax.grid(True)
    
    _save_or_show(fig, fpath)


def plot_matched_spike_trains(gt_spike_trains, 
                              pred_spike_trains, 
                              roas: Optional[List[float]] = None,
                              fpath: Optional[str] = None,
                              spike_count_gt: Optional[List[int]] = None,
                              spike_count_pred: Optional[List[int]] = None) -> None:
    """
    Plot matched spike trains side by side with their metrics.
    
    Args:
        gt_spike_trains: Shape (n_gt, time_steps), ground truth spike trains
        pred_spike_trains: Shape (n_pred, time_steps), predicted spike trains
    """
    gt_spike_trains = _convert_to_numpy(gt_spike_trains)
    pred_spike_trains = _convert_to_numpy(pred_spike_trains)

    T = gt_spike_trains.shape[1]

    n_gt = gt_spike_trains.shape[0]
    n_pred = pred_spike_trains.shape[0]
        
    # Check if there are any valid matches
    if n_gt == 0 or n_pred == 0:
        logging.warning("No matched spikes found")
        return
    
    fig = plt.figure(figsize=(15, 2*n_gt))
    gs = gridspec.GridSpec(n_gt, 2, figure=fig)

    for idx in range(n_gt - 1):
        # Plot ground truth spikes
        ax_gt = fig.add_subplot(gs[idx, 0])
        spike_times = np.where(gt_spike_trains[idx] > 0)[0]
        if len(spike_times) > 0:
            ax_gt.vlines(spike_times, 0, 1, color='red', linewidth=0.8)
        ax_gt.set_ylim(0, 1)
        ax_gt.set_xlim(0, T)
        title = f'Ground Truth {idx+1}\nSpikes: {spike_count_gt[idx]}' if spike_count_gt is not None else f'Ground Truth {idx+1}'
        ax_gt.set_title(title)
        if idx == n_gt - 1:
            ax_gt.set_xlabel('Time (samples)')
        else:
            ax_gt.set_xticks([])

        if idx < n_pred:
            # Plot predicted spikes
            ax_pred = fig.add_subplot(gs[idx, 1])
            spike_times = np.where(pred_spike_trains[idx] > 0)[0]
            if len(spike_times) > 0:
                ax_pred.vlines(spike_times, 0, 1, color='blue', linewidth=0.8)
            ax_pred.set_ylim(0, 1)
            ax_pred.set_xlim(0, T)
            title = f'Predicted {idx+1}\nRoA: {roas[idx]:.3f}   Spikes: {spike_count_pred[idx]}' if roas is not None and spike_count_pred is not None else f'Predicted {idx+1}'
            ax_pred.set_title(title)
            if idx == n_gt - 1:
                ax_pred.set_xlabel('Time (samples)')
            else:
                ax_pred.set_xticks([])
        
    plt.tight_layout()
    _save_or_show(fig, fpath)
