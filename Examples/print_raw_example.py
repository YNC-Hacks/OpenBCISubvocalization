from pyOpenBCI import OpenBCIGanglion

def print_raw(sample):
    print(sample.channels_data)

#Set (daisy = True) to stream 16 ch 
board = OpenBCIGanglion(mac='E0:40:DA:FF:A2:F7')

board.start_stream(print_raw)
