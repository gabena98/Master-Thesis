from datetime import datetime
import pickle
import argparse
import os
import pandas as pd
from datasets.data_labelling import LabelsForData


def mimic_processing(adm_file, dx_file, icd10toicd9_file, single_dx_file, multi_dx_file, out_file):

    adms_df = pd.read_csv(adm_file, compression='gzip', header=0, sep=',')
    dx_df = pd.read_csv(dx_file, compression='gzip', header=0, sep=',')
    icd10cmtoicd9gem_df = pd.read_csv(icd10toicd9_file, header=0, sep=',', quotechar='"')

    icd10cmtoicd9 = {}
    for index, row in icd10cmtoicd9gem_df.iterrows():
        icd10cmtoicd9[row.icd10cm] = row.icd9cm
    label4data = LabelsForData(multi_dx_file, single_dx_file)

    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}
    admDateMap = {}
    # per ogni paziente e ammissione, salvo l'id dell'ammissione e la data di ammissione in due dizionari distinti 
    for index, row in adms_df.iterrows():
        pid = int(row.subject_id)
        admId = int(row.hadm_id)
        admTime = datetime.strptime(row.admittime, '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap:
            pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid] = [admId]

    print('Building admission-dxList mapping')
    admDxMap = {}
    admDxMap_ccs = {}
    admDxMap_ccs_cat1 = {}

    # scorro il dataframe delle diagnoses_icd
    for index, row in dx_df.iterrows():
        admId = int(row.hadm_id)
        dx = row.icd_code.strip()
        if len(dx) == 0:
            continue
        # convert ICD10CM to ICD9CM
        if row.icd_version == 10:
            if dx in icd10cmtoicd9:
                dx = icd10cmtoicd9[dx]
                if dx == 'NoDx':
                    continue
            else:
                continue

        dxStr = 'D_' + dx # codice ICD-9-CM
        dxStr_ccs_single = 'D_' + label4data.code2single_dx[dx] # catergoria CCS 
        dxStr_ccs_cat1 = 'D_' + label4data.code2first_level_dx[dx] # catergoria CCS di primo livello

        if admId in admDxMap:
            admDxMap[admId].append(dxStr)
        else:
            admDxMap[admId] = [dxStr]

        if admId in admDxMap_ccs:
            admDxMap_ccs[admId].append(dxStr_ccs_single)
        else:
            admDxMap_ccs[admId] = [dxStr_ccs_single]

        if admId in admDxMap_ccs_cat1:
            admDxMap_ccs_cat1[admId].append(dxStr_ccs_cat1)
        else:
            admDxMap_ccs_cat1[admId] = [dxStr_ccs_cat1]

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    pidSeqMap_ccs = {}
    pidSeqMap_ccs_cat1 = {}
    for pid, admIdList in pidAdmMap.items(): # id_paziente, lista id_ammissioni
        new_admIdList = []
        for admId in admIdList:
            if admId in admDxMap: # se l'ammissione ha una diagnosi
                new_admIdList.append(admId)
        if len(new_admIdList) < 2: # almeno due ammissioni
            continue
        # print(admIdList)
        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in new_admIdList]) # lista di tuple (data, diagnosi)
        pidSeqMap[pid] = sortedList # dizionario con chiave id_paziente e valore lista di tuple (data, diagnosi)

        sortedList_ccs = sorted([(admDateMap[admId], admDxMap_ccs[admId]) for admId in new_admIdList])
        pidSeqMap_ccs[pid] = sortedList_ccs

        sortedList_ccs_cat1 = sorted([(admDateMap[admId], admDxMap_ccs_cat1[admId]) for admId in new_admIdList])
        pidSeqMap_ccs_cat1[pid] = sortedList_ccs_cat1

    print('Building strSeqs, span label')
    seqs = []
    seqs_span = []
    seqs_intervals = []
    for pid, visits in pidSeqMap.items(): # id_paziente, lista di ricoveri. Ogni ricovero è una lista di tuple: (data, ICD-9-CM)
        seq = []
        spans = []
        intervals = []
        first_time = visits[0][0] # data del primo ricovero
        for i, visit in enumerate(visits):
            current_time = visit[0] # data del ricovero corrente
            interval = (current_time - first_time).days # differenza in giorni tra la data del ricovero corrente e la data del primo ricovero
            first_time = current_time # aggiorna la data del primo ricovero
            seq.append(visit[1]) # aggiunge le diagnosi (codici ICD-9) del ricovero corrente
            span_flag = 0 if interval <= 30 else 1 # se l'intervallo è minore o uguale a 30 giorni, span_flag = 0, altrimenti span_flag = 1
            spans.append(span_flag) # aggiunge span_flag
            intervals.append(interval) # aggiunge l'intervallo
        seqs.append(seq)
        seqs_span.append(spans)
        seqs_intervals.append(intervals)

    print('Building strSeqs for CCS single-level code')
    seqs_ccs = []
    for pid, visits in pidSeqMap_ccs.items(): # id_paziente, lista di tuple (data, diagnosi codici ICD-9-CM)
        seq = []
        for visit in visits:
            seq.append(visit[1])
        seqs_ccs.append(seq)

    print('Converting strSeqs to intSeqs, and making types for ccs single-level code')
    dict_ccs = {}
    newSeqs_ccs = []
    for patient in seqs_ccs: # lista di liste di diagnosi
        newPatient = []
        for visit in patient: # lista di diagnosi
            newVisit = []
            for code in set(visit): # codici di diagnosi ICD-9-CM
                if code in dict_ccs:
                    newVisit.append(dict_ccs[code])
                else:
                    dict_ccs[code] = len(dict_ccs) # assegna un id a ciascun codice di diagnosi
                    newVisit.append(dict_ccs[code]) # aggiunge l'id del codice di diagnosi
            newPatient.append(newVisit)
        newSeqs_ccs.append(newPatient)

    print('Building strSeqs for CCS multi-level first code')
    seqs_ccs_cat1 = []
    for pid, visits in pidSeqMap_ccs_cat1.items(): # id_paziente, lista di tuple (data, diagnosi)
        seq = []
        for visit in visits: # lista di tuple (data, diagnosi)
            seq.append(visit[1]) # diagnosi
        seqs_ccs_cat1.append(seq)

    print('Converting strSeqs to intSeqs, and making types for ccs multi-level first level code')
    dict_ccs_cat1 = {}
    newSeqs_ccs_cat1 = []
    for patient in seqs_ccs_cat1: # per ogni paziente ho liste di diagnosi
        newPatient = []
        for visit in patient: # lista di visite
            newVisit = []
            for code in set(visit): # codici di diagnosi
                if code in dict_ccs_cat1:
                    newVisit.append(dict_ccs_cat1[code])
                else:
                    dict_ccs_cat1[code] = len(dict_ccs_cat1) # assegna un id a ciascun codice di diagnosi
                    newVisit.append(dict_ccs_cat1[code]) # aggiunge l'id del codice di diagnosi
            newPatient.append(newVisit) 
        newSeqs_ccs_cat1.append(newPatient)

    print('Converting seqs to model inputs')
    #pid_list = list(pidSeqMap.keys())
    inputs_all = []
    labels_ccs = []
    intervals_all = []
    labels_current_visit = []
    labels_next_visit = []
    labels_visit_cat1 = []
    vocab_set = {}
    max_visit_len = 0
    max_seqs_len = 0
    truncated_len = 21
    for i, seq in enumerate(seqs): # lista di liste di diagnosi
        length = len(seq)

        if length >= truncated_len: # se la lunghezza della lista di visite è maggiore o uguale a 21
            last_seqs = seq[length-truncated_len:] # prende le ultime 21 visite 
            last_spans = seqs_span[i][length-truncated_len:] # prende le ultime 21 etichette di span   
            last_seq_ccs = newSeqs_ccs[i][length-truncated_len:] # prende le ultime 21 visite con id
            last_seq_ccs_cat1 = newSeqs_ccs_cat1[i][length-truncated_len:] # prende le ultime 21 visite con id
            last_seq_intervals = seqs_intervals[i][length - truncated_len:] # prende gli ultimi 21 intervalli
        else:
            last_seqs = seq
            last_spans = seqs_span[i]
            last_seq_ccs = newSeqs_ccs[i]
            last_seq_ccs_cat1 = newSeqs_ccs_cat1[i]
            last_seq_intervals = seqs_intervals[i]

        valid_seq = last_seqs[:-1] # prende tutte le visite escludendo l'ultima
        label_span = last_spans[-1] # prende l'ultima visita di span
        label_ccs = last_seq_ccs[-1] # prende l'ultima visita con id
        label_current_visit = last_seq_ccs[:-1] # prende tutte le visite escludendo l'ultima
        label_next_visit = last_seq_ccs[1:] # prende tutte le visite escludendo la prima
        valid_intervals = last_seq_intervals[:-1] # prende tutti gli intervalli escludendo l'ultimo

        labels_current_visit.append(label_current_visit)
        labels_next_visit.append(label_next_visit)
        labels_visit_cat1.append(last_seq_ccs_cat1[:-1]) # prende tutte le visite escludendo l'ultima
        inputs_all.append(valid_seq)
        labels_ccs.append((label_ccs))
        intervals_all.append(valid_intervals)

        if len(valid_seq) > max_seqs_len:
            max_seqs_len = len(valid_seq)

        for visit in valid_seq:
            if len(visit) > max_visit_len:
                max_visit_len = len(visit)
            for code in visit:
                if code in vocab_set:
                    vocab_set[code] += 1
                else:
                    vocab_set[code] = 1

    #pickle.dump(pid_list, open(os.path.join(out_file, 'pids.seqs'), 'wb'), -1)
    pickle.dump(inputs_all, open(os.path.join(out_file,'inputs_all.seqs'), 'wb'), -1)
    pickle.dump(intervals_all, open(os.path.join(out_file, 'intervals_all.seqs'), 'wb'), -1)
    pickle.dump(labels_ccs, open(os.path.join(out_file, 'labels_ccs.label'), 'wb'), -1)
    pickle.dump(labels_current_visit, open(os.path.join(out_file, 'labels_current_visit.label'), 'wb'), -1)
    pickle.dump(labels_next_visit, open(os.path.join(out_file, 'labels_next_visit.label'), 'wb'), -1)
    pickle.dump(labels_visit_cat1, open(os.path.join(out_file, 'labels_visit_cat1.label'), 'wb'), -1)
    pickle.dump(dict_ccs, open(os.path.join(out_file, 'ccs_single_level.dict'), 'wb'), -1)
    pickle.dump(dict_ccs_cat1, open(os.path.join(out_file, 'ccs_cat1.dict'), 'wb'), -1)

    pickle.dump(seqs, open(os.path.join(out_file, 'origin.seqs'), 'wb'), -1)
    pickle.dump(newSeqs_ccs, open(os.path.join(out_file, 'origin_ccs.seqs'), 'wb'), -1)
    pickle.dump(newSeqs_ccs_cat1, open(os.path.join(out_file, 'origin_ccs_cat1.seqs'), 'wb'), -1)

    sorted_vocab = {k: v for k, v in sorted(vocab_set.items(), key=lambda item: item[1], reverse=True)} # ordina il vocabolario in base al numero di occorrenze
    outfd = open(os.path.join(out_file, 'vocab.txt'), 'w') # crea un file vocab.txt
    for k, v in sorted_vocab.items(): # per ogni codice di diagnosi
        outfd.write(k + '\n') # scrive il codice di diagnosi
    outfd.close() # chiude il file
    print(max_visit_len, max_seqs_len, len(dict_ccs), len(inputs_all), len(sorted_vocab), len(dict_ccs_cat1)) # stampa la lunghezza massima delle diagnosi, la lunghezza massima delle liste di diagnosi, il numero di diagnosi, il numero di pazienti, il numero di codici di diagnosi, il numero di diagnosi di primo livello


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The output directory where the processed files will be written.")
    parser.add_argument("--admission",
                        type=str,
                        required=True,
                        help="The path of admission file.")
    parser.add_argument("--diagnosis",
                        type=str,
                        required=True,
                        help="The path of diagnosis file.")
    parser.add_argument("--single_level",
                        type=str,
                        required=True,
                        help="The path of CCS Single Level of diagnoses.")
    parser.add_argument("--multi_level",
                        # default=None,
                        type=str,
                        required=True,
                        help="The path of CCS multi-level of diagnoses.")
    parser.add_argument("--icd_convert",
                        # default=None,
                        type=str,
                        required=True,
                        help="The path of ICD10CM to ICD9CM.")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    mimic_processing(args.admission, args.diagnosis, args.icd_convert, args.single_level,
                     args.multi_level, args.output)

