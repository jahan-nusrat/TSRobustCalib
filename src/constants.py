from enum import Enum

class Label_seizure_normal(Enum):
    NORMAL = 0
    SEIZURE = 1
    
SEIZURE_LABEL_MAPPING = {
    "BCKG" : Label_seizure_normal.NORMAL,
    "SEIZ" : Label_seizure_normal.SEIZURE,
    "FNSZ" : Label_seizure_normal.SEIZURE,
    "GNSZ" : Label_seizure_normal.SEIZURE,
    "SPSZ" : Label_seizure_normal.SEIZURE,
    "CPSZ" : Label_seizure_normal.SEIZURE,
    "ABSZ" : Label_seizure_normal.SEIZURE,
    "TNSZ" : Label_seizure_normal.SEIZURE,
    "CNSZ" : Label_seizure_normal.SEIZURE,
    "TCSZ" : Label_seizure_normal.SEIZURE,
    "ATSZ" : Label_seizure_normal.SEIZURE,
    "MYSZ" : Label_seizure_normal.SEIZURE,
    "NESZ" : Label_seizure_normal.SEIZURE
}

ORDERED_CHANNELS = ['EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG A1-LE', 'EEG A2-LE', 
                    'EEG P3-LE', 'EEG P4-LE', 
                    'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 
                    'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE', 'EEG OZ-LE', 'EEG PG1-LE', 'EEG PG2-LE', 
                    'EEG EKG-LE', 'EEG SP2-LE', 'EEG SP1-LE', 'EEG RLC-LE', 'EEG LUC-LE', 'EEG 30-LE', 'EEG T1-LE', 
                    'EEG T2-LE', 'PHOTIC PH']

MONTAGES_NAMES = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "A1-T3",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "T4-A2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2"
]

ORDERED_CHANNELS_REDUCED = [
    'EEG FP1',
    'EEG FP2',
    'EEG F3',
    'EEG F4',
    'EEG C3',
    'EEG C4',
    'EEG P3',
    'EEG P4',
    'EEG O1',
    'EEG O2',
    'EEG F7',
    'EEG F8',
    'EEG T3',
    'EEG T4',
    'EEG T5',
    'EEG T6',
    'EEG FZ',
    'EEG CZ',
    'EEG PZ']

MONTAGES_NAMES_REDUCED = [
    "FP1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",
    "FP2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
    "T3-C3",
    "C3-CZ",
    "CZ-C4",
    "C4-T4",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2"
]

MONTAGES_CHANNELS_IDX = [
    [ORDERED_CHANNELS.index(f"EEG {ch1}-LE"), ORDERED_CHANNELS.index(f"EEG {ch2}-LE")]
    for ch1, ch2 in [montage.split("-") for montage in MONTAGES_NAMES]
]


MONTAGES_CHANNELS_IDX_REDUCED = [
    [ORDERED_CHANNELS.index(f"EEG {ch1}-LE"), ORDERED_CHANNELS.index(f"EEG {ch2}-LE")]
    for ch1, ch2 in [montage.split("-") for montage in MONTAGES_NAMES_REDUCED]
]
