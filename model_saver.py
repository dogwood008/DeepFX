
# coding: utf-8

# In[ ]:


import warnings
import itertools
import rl.callbacks
import numpy as np


# In[ ]:


class ModelSaver(rl.callbacks.TrainEpisodeLogger):
    def __init__(self, filepath, monitor='loss', verbose=1, 
                 save_best_only=True, mode='min', save_weights_only=False,
                 logger=None):
        if filepath is None:
            raise ValueError('Give value to filepath. (Given: %s)' % filepath)
        self.best_monitor_value = None
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_weights_only = save_weights_only
        if mode not in ('min', 'max'):
            raise ValueError("Give 'min' or 'max' to mode. (Given: %s)" % mode)
        self.mode = mode
        self._logger = logger
        
        super().__init__()

    def on_episode_end(self, episode, logs):
        self._logger.warn('========== Model Saver output ==============')
        try:
            monitor_value = float(self._formatted_metrics(episode)[self.monitor])
        except:
            monitor_value = 0.0
        self._logger.warn('%s value: %e' % (self.monitor, monitor_value))
        values = {'episode': episode, self.monitor: monitor_value}
        if not self.save_best_only:
            values['previous_monitor'] = monitor_value
            self._save_model(values)            
        elif self.best_monitor_value is None or self._is_this_episode_improved(monitor_value):
            previous_value = self.best_monitor_value
            self.best_monitor_value = monitor_value
            values['previous_monitor'] = previous_value
            self._save_model(values)
            self._logger.warn('%s %s value: %e' % (self.mode, self.monitor, self.best_monitor_value))
        #except:
        #    self._logger.warn('Not a float value given.')
        self._logger.warn('========== /Model Saver output =============')
        super().on_episode_end(episode, logs)

    def _is_this_episode_improved(self, monitor_value):
        if self.mode == 'min':
            return monitor_value < self.best_monitor_value
        else:
            return monitor_value > self.best_monitor_value
        
    def _save_model(self, kwargs):
        previous_monitor = kwargs['previous_monitor']
        filepath = self.filepath.format_map(kwargs)
        if self.verbose > 0:
            self._logger.warn("Step %05d: model improved\n  from %e\n    to %e,"
                  ' saving model to %s'
                  % (self.step, previous_monitor or 0.0,
                     self.best_monitor_value or 0.0, filepath))
        if self.save_weights_only:
            self.model.save_weights(filepath + '.hdf5', overwrite=True)
            self._logger.warn('Save weights to %s has done.' % filepath)
        else:
            self.model.model.save(filepath + '.h5', overwrite=True)
            self._logger.warn('Save model to %s has done.' % filepath)

    def _formatted_metrics(self, episode):
        # Format all metrics.
        metrics = np.array(self.metrics[episode])
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                try:
                    value = np.nanmean(metrics[:, idx])
                except Warning:
                    if name == 'loss':
                        value = float('inf')
                    else:
                        value = '--'
                metrics_variables += [name, value]
        return dict(itertools.zip_longest(*[iter(metrics_variables)] * 2, fillvalue=""))
        

