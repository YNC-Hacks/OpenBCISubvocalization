from pyOpenBCI import OpenBCIGanglion
from pylsl import StreamInfo, StreamOutlet, ContinuousResolver
import numpy as np
import random
import time

SCALE_FACTOR_EEG = ((4500000)/24/(2**23-1))/1.5 #uV/count
SCALE_FACTOR_AUX = 0.002 / (2**4)


print("Creating LSL stream for EEG. \nName: OpenBCIEEG\nID: OpenBCItestEEG\n")

info_eeg = StreamInfo('OpenBCIEEG', 'EEG', 4, 250, 'float32', 'OpenBCItestEEG')

print("Creating LSL stream for AUX. \nName: OpenBCIAUX\nID: OpenBCItestEEG\n")

info_aux = StreamInfo('OpenBCIAUX', 'AUX', 3, 250, 'float32', 'OpenBCItestAUX')

outlet_eeg = StreamOutlet(info_eeg)
outlet_aux = StreamOutlet(info_aux)

info = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'myuidw43536')
# next make an outlet
outlet = StreamOutlet(info)
markernames = ['Marker', 'Marker', 'Marker', 'Marker', 'Marker', 'Marker']


def lsl_streamers(sample):
    outlet_eeg.push_sample(np.array(sample.channels_data)*SCALE_FACTOR_EEG)
    outlet.push_sample([random.choice(markernames)])
    print(np.array(sample.channels_data)*SCALE_FACTOR_EEG)
    # outlet_aux.push_sample(np.array(sample.aux_data)*SCALE_FACTOR_AUX)

board = OpenBCIGanglion(mac='E0:40:DA:FF:A2:F7')

board.start_stream(lsl_streamers)



# myResol = ContinuousResolver()
# print(myResol.results)


