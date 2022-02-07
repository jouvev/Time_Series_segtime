from mne.datasets import sleep_physionet
from mne.io import concatenate_raws, read_raw_edf


from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import mne

import numpy as np

import matplotlib.pyplot as plt

# #Define the parameters 
# subject = 1  # use data from subject 1
# runs = [6, 10, 14]  # use only hand and feet motor imagery runs

# #Get data and locate in to given path
# files = eegbci.load_data(subject, runs, '../datasets/')


edf = sleep_physionet.age.fetch_data(subjects=[0], recording=[1], path="D:/all/travail/m2_info/AMAL/projet/data")
#np.savetxt(data+name+'-Hypnogram.csv', edf.get_data().T, delimiter=',', header=header)
# files = np.array(edf)

# raws = [read_raw_edf(f, preload=True) for f in files[:,0]]

# raw_obj = concatenate_raws(raws)
raw_train = mne.io.read_raw_edf(edf[0][0])

annot_train = mne.read_annotations(edf[0][1])

raw_train.set_annotations(annot_train, emit_warning=False)

raw_train.plot(start=60, duration=60,
               scalings=dict(eeg=1e-4, resp=1e3, eog=1e-4, emg=1e-7,
                             misc=1e-1))

annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}

# keep last 30-min wake events before sleep and first 30-min wake events after
# sleep and redefine annotations on raw data
annot_train.crop(annot_train[1]['onset'] - 30 * 60,
                 annot_train[-2]['onset'] + 30 * 60)
raw_train.set_annotations(annot_train, emit_warning=False)

events_train, _ = mne.events_from_annotations(
    raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)

# create a new event_id that unifies stages 3 and 4
event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5}

# plot events
fig = mne.viz.plot_events(events_train, event_id=event_id,
                          sfreq=raw_train.info['sfreq'],
                          first_samp=events_train[0, 0])

# keep the color-code for further plotting
stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


epochs_train = mne.Epochs(raw=raw_train, events=events_train,
                          event_id=event_id, tmin=0., tmax=0.5, baseline=None, preload=True)

print(epochs_train)