

#%% import
import os

from scipy.interpolate import interp1d

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import gc
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
from scipy.io import loadmat
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import SE
import warnings
from scipy import signal
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tensorflow.keras import layers

warnings.simplefilter('ignore')


#%% config
# Sub chb24 chb24-summary.txt modified
CFG = {'dataset':       'CHBMIT', # CHBMIT, SWECETHZ  ## Test
       'cross_site':    'CHBMIT', # if CFG['dataset']!=CFG['cross_site'], cross-site epileptic seizure detection  ## Train
       'model':         'CEBT',   # CEBT EEGNet AttBiLSTM CEBTv1 CEBTv2     #引用模型models
       'MinMaxMode':    'Max',
       'resample':      128,
       'seed':          42,
       'start_sub':     0,   # 0（数据人数更多可改为）      1代表ch02作为测试集（只有ch01和ch02）   #抽中哪一个，哪一个数据集作为测试集，其余作为训练集
       'timestep':      0.5, # Ictal timestep 0.5 s
       'permutation':   True,
       'valrate':       0.2,
       'filter_range':  [1*2/128, 60*2/128], # 1-60
       'filter_order':  6,
       'chanrm':        ['ECG', 'VNS', '-', 'LUE-RAE', 'EKG1-EKG2', 'LOC-ROC', 'EKG1-CHIN',
                         'FC1-Ref', 'FC2-Ref', 'FC5-Ref', 'FC6-Ref',
                         'CP1-Ref', 'CP2-Ref', 'CP5-Ref', 'CP6-Ref'], # channels were removed (CHBMIT)
       'CHBMITMax':     32,  # CHBMIT
       'CHBMITMin':     18,  # CHBMIT
       'SWECETHZMax':   128, # SWECETHZ
       'SWECETHZMin':   24,
       'WindowLen':     2,
       'epochs':        100
    ,
       'lr':            1e-3,
       'batchsize':     32,
       'NonMats':       10, # Nonictal hours to test 选择测试集中的几个
       'poststep':      1, # test code, do not change
       'postwindows':   3,
       'mode':          'run', # debug  run
       'trainmax':      3000, # 100 min
       'valmax':        300, # 20 min
       'patience':      10,
       'activation':    'softmax',
       }
CFG['FilePath'] = os.getcwd()
CFG['DataPath'] = os.path.join(CFG['FilePath'], CFG['dataset'])
CFG['DatatempPath'] = CFG['DataPath'] + '_temp'
np.random.seed(CFG['seed'])


#%% load info
def infoloader(CFG):
    dataset = CFG['dataset']
    print('dataset:  ' + CFG['dataset'])
    files = os.listdir(CFG['DataPath'])
    files.sort()
    DatasetDf = pd.DataFrame()
    if dataset=='CHBMIT':
        #DatasetDf = pd.DataFrame()
        for i in range(len(files)):
            SubjectDf = pd.DataFrame()
            SubjectPath = os.path.join(CFG['DataPath'], files[i])
            InfoPath = os.path.join(SubjectPath,  (files[i] + '-summary.txt'))
            with open(InfoPath, "r") as f:
                SubjectInfo = f.read()
            SubjectInfo = SubjectInfo.split('\n')
            Srate = SubjectInfo[0][-6:-3]
            for j in range(len(SubjectInfo)):
                if SubjectInfo[j][:7]=='Channel' and SubjectInfo[j][:8]!='Channels':
                    if SubjectInfo[j-1][0]=='*':
                        SubChannel = list()
                    SubChannel.append(SubjectInfo[j])
                elif SubjectInfo[j][:9]=='File Name':
                    FileName = SubjectInfo[j][11:]
                elif SubjectInfo[j][:26]=='Number of Seizures in File':
                    NumSeiz = int(SubjectInfo[j][27:])
                    SeizTime = np.zeros((1, 2))
                    for seiz in range(NumSeiz):
                        seizstart = SubjectInfo[j+(seiz*2+1)][SubjectInfo[j+(seiz*2+1)].find(':')+2: -8]
                        seizend   = SubjectInfo[j+(seiz*2+2)][SubjectInfo[j+(seiz*2+2)].find(':')+2: -8]
                        SeizTime = np.vstack((SeizTime, (int(seizstart), int(seizend))))
                    SeizTime = SeizTime[1:, :]
                    SeizTimeAll = sum(SeizTime[:, 1]-SeizTime[:, 0])
                    SubjectDf = SubjectDf.append({'Sub': files[i], 'SubjectPath': SubjectPath, 'FileName': FileName,
                                                  'Srate': Srate, 'NumSeiz': NumSeiz, 'SeizTime': SeizTime,
                                                  'SeizTimeAll': SeizTimeAll, 'Cha': SubChannel,
                                                  }, ignore_index=True)
            IctalDf = SubjectDf[SubjectDf['NumSeiz']>0].reset_index(drop=True)
            NonictalDf = SubjectDf[SubjectDf['NumSeiz']==0].reset_index(drop=True)
            if len(IctalDf)>0:
                IctalTrain = max(len(IctalDf) - max(np.around(len(IctalDf)*CFG['valrate']), 1), 1)
                IctalDf.loc[:IctalTrain, 'TrainVal'] = 'train'
                IctalDf.loc[IctalTrain:, 'TrainVal'] = 'val'
            if len(NonictalDf)>0:
                NonictalTrain = max(len(NonictalDf) - max(np.around(len(NonictalDf)*CFG['valrate']), 1), 1)
                NonictalDf.loc[:NonictalTrain, 'TrainVal'] = 'train'
                NonictalDf.loc[NonictalTrain:, 'TrainVal'] = 'val'
            DatasetDf = DatasetDf.append(IctalDf, ignore_index=True)
            DatasetDf = DatasetDf.append(NonictalDf, ignore_index=True)
        CFG['Subjects'] = files
    elif dataset=='SWECETHZ':
        #DatasetDf = pd.DataFrame()
        for i in range(len(files)-1):
            SubjectIctalDf = pd.DataFrame()
            SubjectNonictalDf = pd.DataFrame()
            SubjectPath = os.path.join(CFG['DataPath'], files[i])
            InfoPath = os.path.join(CFG['DataPath'],  files[len(files)-1], files[i]+'_info.mat')
            Info = loadmat(InfoPath)
            Mats = os.listdir(SubjectPath)
            Mats.sort()
            for j in range(len(Info['seizure_begin'])):
                SeizTime = np.zeros((1, 2))
                SeizBegin = Info['seizure_begin'][j]
                SeizEnd   = Info['seizure_end'][j]
                Hour = int(SeizBegin/3600)+1
                SeizBegin = SeizBegin[0]%3600
                SeizEnd   = SeizEnd[0]%3600
                if SeizEnd>SeizBegin:
                    SeizTime = np.vstack((SeizTime, (SeizBegin, SeizEnd)))
                else:
                    SeizTime = np.vstack((SeizTime, (SeizBegin, 3600)))
                    SeizTime = np.vstack((SeizTime, (0, SeizEnd)))
                SeizTime = SeizTime[1:, :]
                SubjectIctalDf = SubjectIctalDf.append({'Sub': files[i], 'SubjectPath': SubjectPath,
                                                        'FileName': files[i]+'_'+str(Hour)+'h.mat',
                                                        'Srate': Info['fs'][0][0], 'NumSeiz': 1,
                                                        'SeizTime': SeizTime[0, :],
                                                        'SeizTimeAll': SeizTime[0, 1]-SeizTime[0, 0],
                                                        }, ignore_index=True)
                if (SeizTime.shape[0]>1) and (SeizTime[1, 1]-SeizTime[1, 0]>CFG['WindowLen']):
                    SubjectIctalDf = SubjectIctalDf.append({'Sub': files[i], 'SubjectPath': SubjectPath,
                                                            'FileName': files[i]+'_'+str(Hour+1)+'h.mat',
                                                            'Srate': Info['fs'][0][0], 'NumSeiz': 1,
                                                            'SeizTime': SeizTime[1, :],
                                                            'SeizTimeAll': SeizTime[1, 1]-SeizTime[1, 0],
                                                            }, ignore_index=True)
            for j in range(len(Mats)):
                if not Mats[j] in list(SubjectIctalDf['FileName']):
                    SubjectNonictalDf = SubjectNonictalDf.append({'Sub': files[i], 'SubjectPath': SubjectPath,
                                                                  'FileName': Mats[j],
                                                                  'Srate': Info['fs'][0][0], 'NumSeiz': 0,
                                                                  'SeizTimeAll': 0
                                                                  }, ignore_index=True)
            print(files[i] + '  Ictal mats:  ' + str(len(SubjectIctalDf)) + '  Nonictal mats:  ' + str(len(SubjectNonictalDf)))
            r = np.random.permutation(len(SubjectNonictalDf))
            SubjectNonictalDf = SubjectNonictalDf.loc[r].reset_index(drop=True)
            SubjectNonictalDf = SubjectNonictalDf[:CFG['NonMats']]          
            if len(SubjectIctalDf)>0:
                IctalTrain = max(len(SubjectIctalDf) - max(np.around(len(SubjectIctalDf)*CFG['valrate']), 1), 1)
                SubjectIctalDf.loc[:IctalTrain, 'TrainVal'] = 'train'
                SubjectIctalDf.loc[IctalTrain:, 'TrainVal'] = 'val'
            if len(SubjectNonictalDf)>0:
                NonictalTrain = max(len(SubjectNonictalDf) - max(np.around(len(SubjectNonictalDf)*CFG['valrate']), 1), 1)
                SubjectNonictalDf.loc[:NonictalTrain, 'TrainVal'] = 'train'
                SubjectNonictalDf.loc[NonictalTrain:, 'TrainVal'] = 'val'
            DatasetDf = DatasetDf.append(SubjectIctalDf, ignore_index=True)
            DatasetDf = DatasetDf.append(SubjectNonictalDf, ignore_index=True)
        CFG['Subjects'] = files[:-1]
    return CFG, DatasetDf


#%% select data
def chanrm(EdfChan, CFG):
    EdfChanBool = list()
    for k in range(len(EdfChan)):
        EdfChanTemp = EdfChan[k]
        EdfChanTemp = EdfChanTemp[EdfChanTemp.find(':')+2:]
        if EdfChanTemp in CFG['chanrm']:
            print('channels rm:  ' + EdfChanTemp)
            EdfChanBool.append(False)
        else:
            EdfChanBool.append(True)
    return EdfChanBool

def preprocess(data):
    b, a = signal.butter(CFG['filter_order'], CFG['filter_range'], 'bandpass')
    data = signal.filtfilt(b, a, data, axis=1)
    return data

def dataloader(IctalTrainDf, NonictalTrainDf, CFG, datatype='train'):
    global reshapetemp1
    dataset = CFG['dataset']
    TrainData  = np.zeros((1, CFG[dataset+'Max'], CFG['WindowLen']*CFG['resample']), dtype=np.float32)
    TrainLabel = np.zeros(1, dtype=np.float32)
    for j in range(len(IctalTrainDf)):
        print(IctalTrainDf.loc[j, 'FileName'])
        if dataset=='CHBMIT':
            EdfChanBool = chanrm(IctalTrainDf.loc[j, 'Cha'], CFG)
            data = read_raw_edf(os.path.join(IctalTrainDf.loc[j, 'SubjectPath'], IctalTrainDf.loc[j, 'FileName']), preload=True)
            data = data.get_data()
            data = data[EdfChanBool, :]
        elif dataset=='SWECETHZ':
            data = loadmat(os.path.join(IctalTrainDf.loc[j, 'SubjectPath'], IctalTrainDf.loc[j, 'FileName']))
            data = data['EEG']
        data = preprocess(data)
        if int(IctalTrainDf.loc[j, 'Srate'])>CFG['resample']:
            data = signal.resample_poly(data.T, CFG['resample'], int(IctalTrainDf.loc[j, 'Srate'])).T
        scale_mean = np.mean(data, axis=1)
        scale_std  = np.std(data, axis=1)
        data = ((data.T - scale_mean)/scale_std).T
        for k in range(int(IctalTrainDf.loc[j, 'NumSeiz'])):
            if len(IctalTrainDf.loc[j, 'SeizTime'].shape)==1:
                datatemp = data[:, int(IctalTrainDf.loc[j, 'SeizTime'][0]*CFG['resample']):
                                int(IctalTrainDf.loc[j, 'SeizTime'][1]*CFG['resample'])]
            else:
                datatemp = data[:, int(IctalTrainDf.loc[j, 'SeizTime'][k,0]*CFG['resample']):
                                int(IctalTrainDf.loc[j, 'SeizTime'][k,1]*CFG['resample'])]
            if datatype=='train':
                index = np.arange(0, datatemp.shape[1], CFG['timestep']*CFG['resample'])
                index = index[:int((datatemp.shape[1]-CFG['WindowLen']*CFG['resample'])/CFG['resample']/CFG['timestep']+1)]
            elif datatype=='val':
                index = np.arange(0, data.shape[1], CFG['WindowLen']*CFG['resample'])
                index = index[:int((datatemp.shape[1]-CFG['WindowLen']*CFG['resample'])/CFG['resample']/CFG['WindowLen']+1)]
            reshapetemp = np.zeros((len(index), CFG[dataset+'Max'], CFG['WindowLen']*CFG['resample']), dtype=np.float32)
            for iStep in range(len(index)):
                reshapetemp[iStep, :datatemp.shape[0], :] = datatemp[:, int(index[iStep]):int(index[iStep]+CFG['WindowLen']*CFG['resample'])]

                reshapetemp1 = np.reshape(reshapetemp, (-1, 32))
                #print(reshapetemp1.shape)
                # step=256
                # reshapetemp=[reshapetemp[i:i+step] for i in range(0,len(reshapetemp),step)]
                reshapetemp1=reshapetemp1[0:2560]
                print(reshapetemp1.shape)
                # np.savetxt('shuju.csv', reshapetemp1, delimiter=',')
                #print('reshapetemp[0:255]\n',reshapetemp[0:255])
                #np.savetxt('new2.csv', reshapetemp1, delimiter=',')



                #plt.imshow(reshapetemp,cmap='hot',interpolation='nearest')
                #plt.show()

            #raw.plot_psd(fmax=60)
            # reshapetemp.plot(duration=2, nchannels=32)
            # plt.show()

            TrainData = np.vstack((TrainData, reshapetemp))
            TrainLabel = np.hstack((TrainLabel, np.ones(reshapetemp.shape[0], dtype=np.float32)))
    if len(NonictalTrainDf)>0:
        NonictalTrainDf['NonictalTimeAll'] = int(TrainData.shape[0]*CFG['WindowLen']/len(NonictalTrainDf))
        for j in range(len(NonictalTrainDf)):
            print(NonictalTrainDf.loc[j, 'FileName'])
            if dataset=='CHBMIT':
                EdfChanBool = chanrm(NonictalTrainDf.loc[j, 'Cha'], CFG)
                data = read_raw_edf(os.path.join(NonictalTrainDf.loc[j, 'SubjectPath'], NonictalTrainDf.loc[j, 'FileName']), preload=True)
                data = data.get_data()
                data = data[EdfChanBool, :]
            elif dataset=='SWECETHZ':
                data = loadmat(os.path.join(NonictalTrainDf.loc[j, 'SubjectPath'], NonictalTrainDf.loc[j, 'FileName']))
                data = data['EEG']
            
            data = preprocess(data)
            if int(NonictalTrainDf.loc[j, 'Srate'])>CFG['resample']:
                data = signal.resample_poly(data.T, CFG['resample'], int(NonictalTrainDf.loc[j, 'Srate'])).T
            scale_mean = np.mean(data, axis=1)
            scale_std  = np.std(data, axis=1)
            data = ((data.T - scale_mean)/scale_std).T
            data = data[:, int(data.shape[1]/2):min((int(data.shape[1]/CFG['resample'])*CFG['resample'], (int(data.shape[1]/2)+NonictalTrainDf.loc[j, 'NonictalTimeAll']*CFG['resample'])))]
            
            index = np.arange(0, data.shape[1], CFG['WindowLen']*CFG['resample'])
            index = index[:int((data.shape[1]-CFG['WindowLen']*CFG['resample'])/CFG['resample']/CFG['WindowLen']+1)]
            reshapetemp = np.zeros((len(index), CFG[dataset+'Max'], CFG['WindowLen']*CFG['resample']), dtype=np.float32)
            for iStep in range(len(index)):
                reshapetemp[iStep, :data.shape[0], :] = data[:, int(index[iStep]):int(index[iStep]+CFG['WindowLen']*CFG['resample'])]

                reshapetemp1 = np.reshape(reshapetemp, (-1, 32))
                #print(reshapetemp1.shape)
                reshapetemp1 = reshapetemp1[0:2560]
                print(reshapetemp1.shape)
                #print('reshapetemp[0:255]\n', reshapetemp[0:255])
                # np.savetxt('shuju.csv', reshapetemp1, delimiter=',')

                #plt.imshow(reshapetemp, cmap='hot', interpolation='nearest')
                #plt.show()

            #raw.plot_psd(fmax=50)
            #raw.plot(duration=2, nchannels=32)

            TrainData = np.vstack((TrainData, reshapetemp))
            TrainLabel = np.hstack((TrainLabel, np.zeros(reshapetemp.shape[0], dtype=np.float32)))
    TrainData  = TrainData[1:, :, :]
    #print(TrainData.shape)
    TrainLabel = TrainLabel[1:]
    return TrainData, TrainLabel

def dataselecter(DatasetDf, CFG):
    if not os.path.exists(CFG['DatatempPath']):
        os.mkdir(CFG['DatatempPath'])
    for i in range(len(CFG['Subjects'])):
        SubDf = DatasetDf[DatasetDf['Sub']==CFG['Subjects'][i]].reset_index(drop=True)
        IctalTrainDf = SubDf[(SubDf['TrainVal']=='train')*(SubDf['NumSeiz']>0)].reset_index(drop=True)
        NonictalTrainDf = SubDf[(SubDf['TrainVal']=='train')*(SubDf['NumSeiz']==0)].reset_index(drop=True)
        TrainData, TrainLabel = dataloader(IctalTrainDf, NonictalTrainDf, CFG, 'train')
        if not os.path.exists(os.path.join(CFG['DatatempPath'], CFG['Subjects'][i])):
            os.mkdir(os.path.join(CFG['DatatempPath'], CFG['Subjects'][i]))
        np.save(os.path.join(CFG['DatatempPath'], CFG['Subjects'][i], 'TrainData.npy'), TrainData)
        np.save(os.path.join(CFG['DatatempPath'], CFG['Subjects'][i], 'TrainLabel.npy'), TrainLabel)
        del TrainData, TrainLabel
        gc.collect()
        IctalValDf = SubDf[(SubDf['TrainVal']=='val')*(SubDf['NumSeiz']>0)].reset_index(drop=True)
        NonictalValDf = SubDf[(SubDf['TrainVal']=='val')*(SubDf['NumSeiz']==0)].reset_index(drop=True)
        if len(IctalValDf)>0:
            ValData, ValLabel = dataloader(IctalValDf, NonictalValDf, CFG, 'val')
            if not os.path.exists(os.path.join(CFG['DatatempPath'], CFG['Subjects'][i])):
                os.mkdir(os.path.join(CFG['DatatempPath'], CFG['Subjects'][i]))
            np.save(os.path.join(CFG['DatatempPath'], CFG['Subjects'][i], 'ValData.npy'), ValData)
            np.save(os.path.join(CFG['DatatempPath'], CFG['Subjects'][i], 'ValLabel.npy'), ValLabel)
            del ValData, ValLabel
            gc.collect()


#%% train
def dataassemble(CFG, Subject):
    TrainData  = np.zeros((1, CFG[CFG['dataset']+CFG['MinMaxMode']], CFG['WindowLen']*CFG['resample']), dtype=np.float32)
    ValData    = np.zeros((1, CFG[CFG['dataset']+CFG['MinMaxMode']], CFG['WindowLen']*CFG['resample']), dtype=np.float32)
    TrainLabel = np.zeros(1, dtype=np.float32)
    ValLabel   = np.zeros(1, dtype=np.float32)
    for i in range(len(CFG['Subjects'])):
        if CFG['Subjects'][i]!=Subject:
            DataPath = os.path.join(CFG['DatatempPath'], CFG['Subjects'][i])
            print(DataPath)
            datatrain  = np.load(os.path.join(DataPath, 'TrainData.npy'))
            labeltrain = np.load(os.path.join(DataPath, 'TrainLabel.npy'))
            dataval    = np.load(os.path.join(DataPath, 'ValData.npy'))
            labelval   = np.load(os.path.join(DataPath, 'ValLabel.npy'))

            r = np.random.permutation(datatrain.shape[0])
            datatrain = datatrain[r, :, :]
            labeltrain = labeltrain[r]
            datatrain = datatrain[:CFG['trainmax'], :, :]
            labeltrain = labeltrain[:CFG['trainmax']]

            r = np.random.permutation(dataval.shape[0])
            dataval = dataval[r, :, :]
            labelval = labelval[r]
            dataval = dataval[:CFG['valmax'], :, :]
            labelval = labelval[:CFG['valmax']]

            TrainData  = np.vstack((TrainData, datatrain[:, :CFG[CFG['dataset']+CFG['MinMaxMode']], :]))
            TrainLabel = np.hstack((TrainLabel, labeltrain))
            ValData    = np.vstack((ValData, dataval[:, :CFG[CFG['dataset']+CFG['MinMaxMode']], :]))
            ValLabel   = np.hstack((ValLabel, labelval))
    TrainData  = TrainData[1:, :, :]
    TrainLabel = TrainLabel[1:]

    r = np.random.permutation(TrainData.shape[0])
    TrainData  = TrainData[r, :, :]
    TrainLabel = TrainLabel[r]
    ValData    = ValData[1:, :, :]
    ValLabel   = ValLabel[1:]

    r = np.random.permutation(ValData.shape[0])
    ValData  = ValData[r, :, :]
    ValLabel = ValLabel[r]
    TrainDataIcal    = TrainData[TrainLabel==1, :, :]
    TrainDataNonical = TrainData[TrainLabel==0, :, :]
    del TrainData
    gc.collect()
    ValDataIcal      = ValData[ValLabel==1, :, :]
    ValDataNonical   = ValData[ValLabel==0, :, :]
    del ValData
    gc.collect()
    TrainDataIcal    = TrainDataIcal[:min(TrainDataIcal.shape[0], TrainDataNonical.shape[0]), :, :]
    TrainDataNonical = TrainDataNonical[:min(TrainDataIcal.shape[0], TrainDataNonical.shape[0]), :, :]
    ValDataIcal      = ValDataIcal[:min(ValDataIcal.shape[0], ValDataNonical.shape[0]), :, :]
    ValDataNonical   = ValDataNonical[:min(ValDataIcal.shape[0], ValDataNonical.shape[0]), :, :]
    TrainData = np.vstack((TrainDataIcal, TrainDataNonical))
    TrainLabel = np.hstack((np.ones(TrainDataIcal.shape[0], dtype=np.float32), np.zeros(TrainDataNonical.shape[0], dtype=np.float32)))
    del TrainDataIcal, TrainDataNonical
    gc.collect()

    r = np.random.permutation(TrainData.shape[0])
    TrainData  = TrainData[r, :, :]
    TrainLabel = TrainLabel[r]
    ValData = np.vstack((ValDataIcal, ValDataNonical))
    ValLabel = np.hstack((np.ones(ValDataIcal.shape[0], dtype=np.float32), np.zeros(ValDataNonical.shape[0], dtype=np.float32)))
    del ValDataIcal, ValDataNonical
    gc.collect()

    r = np.random.permutation(ValData.shape[0])
    ValData  = ValData[r, :, :]
    ValLabel = ValLabel[r]
    if (CFG['dataset'] != CFG['cross_site']):
        chans_max = max(CFG['CHBMIT'+CFG['MinMaxMode']], CFG['SWECETHZ'+CFG['MinMaxMode']])
        if TrainData.shape[1]<chans_max:
            TrainData = np.hstack((TrainData, np.zeros((TrainData.shape[0], (chans_max-TrainData.shape[1]), TrainData.shape[2]), dtype=np.float32)))
            ValData = np.hstack((ValData, np.zeros((ValData.shape[0], (chans_max-ValData.shape[1]), ValData.shape[2]), dtype=np.float32)))
    if CFG['permutation']:
        for i in range(TrainData.shape[0]):
            permutation_tmp = np.random.permutation(TrainData.shape[1])
            TrainData[i, :, :] = TrainData[i, permutation_tmp, :]
        for i in range(ValData.shape[0]):
            permutation_tmp = np.random.permutation(ValData.shape[1])
            ValData[i, :, :] = ValData[i, permutation_tmp, :]
    return TrainData, TrainLabel, ValData, ValLabel


def ModelTrain(CFG):
    if CFG['mode']=='debug':
        CFG['start_sub'] = 0
        SubNum = 1
    elif CFG['mode']=='run':
        SubNum = len(CFG['Subjects'])
    for i in range(CFG['start_sub'], SubNum):#不是从第一个数据集开始遍历，从第二个开始
        Subject = CFG['Subjects'][i]
        if (CFG['dataset'] != CFG['cross_site']):
            model = models.model_select(CFG['model'], max(CFG['CHBMIT'+CFG['MinMaxMode']], CFG['SWECETHZ'+CFG['MinMaxMode']]), CFG['resample']*CFG['WindowLen'], 2, activation='softmax')
        else:
            model = models.model_select(CFG['model'], CFG[CFG['dataset'] + CFG['MinMaxMode']], CFG['resample']*CFG['WindowLen'], 2, activation='softmax')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CFG['lr']), loss='binary_crossentropy', metrics=['accuracy'])
        if CFG['permutation']:
            model_save_path = CFG['model'] + '_' + CFG['MinMaxMode'] + '_Permutation.h5'
        else:
            model_save_path = CFG['model'] + '_' + CFG['MinMaxMode'] + '_NonPermutat.h5'
        callbacks_list = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=8),
                          tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(CFG['DatatempPath'], Subject, model_save_path),
                                                             monitor='val_loss', mode='min',
                                                             save_weights_only=True, save_best_only=True),
                          tf.keras.callbacks.EarlyStopping(patience=CFG['patience'], monitor='val_loss', mode='min')]

        TrainData, TrainLabel, ValData, ValLabel = dataassemble(CFG, Subject)
        TrainData = np.nan_to_num(TrainData) #使用0代替数组x中的nan元素
        ValData = np.nan_to_num(ValData)
        print('Train Data Shape:' + str(TrainData.shape))
        print('Valid Data Shape:' + str(ValData.shape))

        TrainLabel = tf.keras.utils.to_categorical(TrainLabel)
        ValLabel   = tf.keras.utils.to_categorical(ValLabel)
        history = model.fit(TrainData, TrainLabel, batch_size=CFG['batchsize'], epochs=CFG['epochs'],
                            verbose=1, callbacks=callbacks_list, validation_data=(ValData, ValLabel))
        del TrainData, TrainLabel, ValData, ValLabel
        gc.collect()
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Train_Acc')
        plt.plot(epochs, val_acc, 'b', label='Val_Acc')
        plt.title('Train and Val Acc', fontsize=20)
        plt.legend()               
        plt.savefig(os.path.join(CFG['DatatempPath'], Subject, 'Acc.pdf'))
        plt.close()
        plt.plot(epochs, loss, 'bo', label='Train_Loss')
        plt.plot(epochs, val_loss, 'b', label='Val_Loss')
        plt.title('Train and Val Loss', fontsize=20)
        plt.legend()
        plt.savefig(os.path.join(CFG['DatatempPath'], Subject, 'Loss.pdf'))
        plt.close()


#%% test
def ModelTest(CFG):
    if (CFG['dataset'] != CFG['cross_site']):
        model = models.model_select(CFG['model'], max(CFG['SWECETHZ' + CFG['MinMaxMode']], CFG['CHBMIT' + CFG['MinMaxMode']]), CFG['resample']*CFG['WindowLen'], 2, activation='softmax')
    else:
        model = models.model_select(CFG['model'], CFG[CFG['dataset'] + CFG['MinMaxMode']], CFG['resample']*CFG['WindowLen'], 2, activation='softmax')
    if CFG['mode']=='debug':
        SubNum = 1
    elif CFG['mode']=='run':
        SubNum = len(CFG['Subjects'])
    for i in range(CFG['start_sub'], SubNum):
        print(CFG['Subjects'][i] + '  testing...')
        if CFG['permutation']:
            model_save_path = CFG['model'] + '_' + CFG['MinMaxMode'] + '_Permutation.h5'
        else:
            model_save_path = CFG['model'] + '_' + CFG['MinMaxMode'] + '_NonPermutat.h5'
        if (CFG['dataset'] != CFG['cross_site']):
            if CFG['cross_site']=='SWECETHZ':
                model.load_weights(os.path.join(CFG['FilePath'], (CFG['cross_site'] + '_temp'), 'ID18', model_save_path))
            elif CFG['cross_site']=='CHBMIT':
                model.load_weights(os.path.join(CFG['FilePath'], (CFG['cross_site'] + '_temp'), 'chb24', model_save_path))
        else:
            model.load_weights(os.path.join(CFG['DatatempPath'], CFG['Subjects'][i], model_save_path))
        SubDf = DatasetDf[DatasetDf['Sub']==CFG['Subjects'][i]].reset_index(drop=True)
        NonictalTime = 0 
        LabelReal = np.zeros(1)
        LabelPre  = np.zeros((1, 2))
        NonictalDf = SubDf[SubDf['NumSeiz']==0].reset_index(drop=True)
        IctalDf    = SubDf[SubDf['NumSeiz']>0].reset_index(drop=True)
        if len(NonictalDf)>0:
            if CFG['dataset']=='CHBMIT':
                NonictalDf = NonictalDf.loc[np.random.permutation(len(NonictalDf))].reset_index(drop=True)
            for j in range(len(NonictalDf)):
                print(NonictalDf.loc[j, 'FileName'])
                if CFG['dataset']=='CHBMIT':
                    EdfChanBool = chanrm(NonictalDf.loc[j, 'Cha'], CFG)
                    data = read_raw_edf(os.path.join(NonictalDf.loc[j, 'SubjectPath'], NonictalDf.loc[j, 'FileName']), preload=True)
                    data = data.get_data()
                    data = data[EdfChanBool, :]
                elif CFG['dataset']=='SWECETHZ':
                    data = loadmat(os.path.join(NonictalDf.loc[j, 'SubjectPath'], NonictalDf.loc[j, 'FileName']))
                    data = data['EEG']
                data = preprocess(data)
                if int(NonictalDf.loc[j, 'Srate'])>CFG['resample']:
                    data = signal.resample_poly(data.T, CFG['resample'], int(NonictalDf.loc[j, 'Srate'])).T
                scale_mean = np.mean(data, axis=1)
                scale_std  = np.std(data, axis=1)
                data = ((data.T - scale_mean)/scale_std).T
                NonictalTime = NonictalTime + data.shape[1]/CFG['resample']
                index = np.arange(0, data.shape[1], CFG['poststep']*CFG['resample'])
                index = index[:int((data.shape[1]-CFG['WindowLen']*CFG['resample'])/CFG['resample']/CFG['poststep']+1)]
                reshapetemp = np.zeros((len(index), CFG[CFG['dataset']+CFG['MinMaxMode']], CFG['WindowLen']*CFG['resample']), dtype=np.float32)
                for iStep in range(len(index)):
                    reshapetemp[iStep, :min(data.shape[0], CFG[CFG['dataset']+CFG['MinMaxMode']]), :] = data[:min(data.shape[0], CFG[CFG['dataset']+CFG['MinMaxMode']]),
                                                                                                             int(index[iStep]):int(index[iStep]+CFG['WindowLen']*CFG['resample'])]
                del data
                gc.collect()
                LabelReal = np.hstack((LabelReal, np.zeros(reshapetemp.shape[0])))
                if (CFG['dataset'] != CFG['cross_site']):
                    chans_max = max(CFG['CHBMIT'+CFG['MinMaxMode']], CFG['SWECETHZ'+CFG['MinMaxMode']])
                    if reshapetemp.shape[1]<chans_max:
                        reshapetemp = np.hstack((reshapetemp, np.zeros((reshapetemp.shape[0], (chans_max-reshapetemp.shape[1]), reshapetemp.shape[2]), dtype=np.float32)))
                LabelPreTemp = model.predict(reshapetemp)


                del reshapetemp
                LabelPre = np.vstack((LabelPre, LabelPreTemp))
                if NonictalTime>CFG['NonMats']*3600:
                    break
        if len(IctalDf)>0:
            for j in range(len(IctalDf)):
                print(IctalDf.loc[j, 'FileName'])
                if CFG['dataset']=='CHBMIT':
                    EdfChanBool = chanrm(IctalDf.loc[j, 'Cha'], CFG)
                    data = read_raw_edf(os.path.join(IctalDf.loc[j, 'SubjectPath'], IctalDf.loc[j, 'FileName']), preload=True)
                    data = data.get_data()
                    data = data[EdfChanBool, :]
                elif CFG['dataset']=='SWECETHZ':
                    data = loadmat(os.path.join(IctalDf.loc[j, 'SubjectPath'], IctalDf.loc[j, 'FileName']))
                    data = data['EEG']
                data = preprocess(data)
                if int(IctalDf.loc[j, 'Srate'])>CFG['resample']:
                    data = signal.resample_poly(data.T, CFG['resample'], int(IctalDf.loc[j, 'Srate'])).T
                scale_mean = np.mean(data, axis=1)
                scale_std  = np.std(data, axis=1)
                data = ((data.T - scale_mean)/scale_std).T
                index = np.arange(0, data.shape[1], CFG['poststep']*CFG['resample'])
                index = index[:int((data.shape[1]-CFG['WindowLen']*CFG['resample'])/CFG['resample']/CFG['poststep']+1)]
                reshapetemp = np.zeros((len(index), CFG[CFG['dataset']+CFG['MinMaxMode']], CFG['WindowLen']*CFG['resample']), dtype=np.float32)
                for iStep in range(len(index)):
                    reshapetemp[iStep, :min(data.shape[0], CFG[CFG['dataset']+CFG['MinMaxMode']]), :] = data[:min(data.shape[0], CFG[CFG['dataset']+CFG['MinMaxMode']]),
                                                                                                             int(index[iStep]):int(index[iStep]+CFG['WindowLen']*CFG['resample'])]
                del data
                gc.collect()
                LabelRealTemp = np.zeros(reshapetemp.shape[0])
                if CFG['dataset']=='CHBMIT':
                    for k in range(round(IctalDf.loc[j, 'NumSeiz'])):
                        LabelRealTemp[int(IctalDf.loc[j, 'SeizTime'][k, 0]):int(IctalDf.loc[j, 'SeizTime'][k, 1])] = 1
                elif CFG['dataset']=='SWECETHZ':
                    LabelRealTemp[int(IctalDf.loc[j, 'SeizTime'][0]):int(IctalDf.loc[j, 'SeizTime'][1])] = 1
                if (NonictalTime<CFG['NonMats']*3600) and CFG['dataset']=='CHBMIT':
                    NonictalTime = NonictalTime + reshapetemp.shape[0]
                    LabelReal = np.hstack((LabelReal, LabelRealTemp))
                    if (CFG['dataset'] != CFG['cross_site']):
                        chans_max = max(CFG['CHBMIT'+CFG['MinMaxMode']], CFG['SWECETHZ'+CFG['MinMaxMode']])
                        if reshapetemp.shape[1]<chans_max:
                            reshapetemp = np.hstack((reshapetemp, np.zeros((reshapetemp.shape[0], (chans_max-reshapetemp.shape[1]), reshapetemp.shape[2]), dtype=np.float32)))
                    LabelPreTemp = model.predict(reshapetemp)


                else:
                    if (CFG['dataset'] != CFG['cross_site']):
                        chans_max = max(CFG['CHBMIT'+CFG['MinMaxMode']], CFG['SWECETHZ'+CFG['MinMaxMode']])
                        if reshapetemp.shape[1]<chans_max:
                            reshapetemp = np.hstack((reshapetemp, np.zeros((reshapetemp.shape[0], (chans_max-reshapetemp.shape[1]), reshapetemp.shape[2]), dtype=np.float32)))
                    LabelPreTemp = model.predict(reshapetemp)


                    del reshapetemp
                    for k in range(len(LabelRealTemp)):
                        if k==0 and LabelRealTemp[k]==1:
                            onset = k
                        elif k>0 and (LabelRealTemp[k]-LabelRealTemp[k-1])>0:
                            onset = max(k-10, 0)
                        if (k<(len(LabelRealTemp)-1) and (LabelRealTemp[k]-LabelRealTemp[k-1])<0) or (k==(len(LabelRealTemp)-1) and LabelRealTemp[k]==1):
                            offset = k
                            if (k==(len(LabelRealTemp)-1) and LabelRealTemp[k]==1):
                                offset = offset + 1
                            LabelReal = np.hstack((LabelReal, LabelRealTemp[onset:offset]))
                            LabelPre = np.vstack((LabelPre, LabelPreTemp[onset:offset]))
                gc.collect()
        LabelReal = LabelReal[1:]
        LabelPre  = LabelPre[1:, :]
        np.save(os.path.join(CFG['DatatempPath'], CFG['Subjects'][i], 'LabelReal.npy'), LabelReal)
        np.save(os.path.join(CFG['DatatempPath'], CFG['Subjects'][i], 'LabelPre.npy'), LabelPre)


#%% postprocess
def Postprocess(CFG):
    ResultsDf = pd.DataFrame()
    if CFG['mode']=='debug':
        SubNum = 1
    elif CFG['mode']=='run':
        SubNum = len(CFG['Subjects'])
    for i in range(CFG['start_sub'], SubNum):
        print(CFG['Subjects'][i])
        LabelReal = np.load(os.path.join(CFG['DatatempPath'], CFG['Subjects'][i], 'LabelReal.npy'))
        LabelPre  = np.load(os.path.join(CFG['DatatempPath'], CFG['Subjects'][i], 'LabelPre.npy'))
        LabelPre2 = np.argmax(LabelPre, axis=1)
        Sen1 = sum(LabelPre2[LabelReal==1])/len(LabelPre2[LabelReal==1])
        Spe  = 1 - sum(LabelPre2[LabelReal==0])/len(LabelPre2[LabelReal==0])
        Acc  = sum(LabelPre2==LabelReal)/len(LabelReal)
        Auc = roc_auc_score(LabelReal, LabelPre[:, 1]);
        F1Binary = f1_score(LabelReal, LabelPre2, average='binary')
        F1Weighted = f1_score(LabelReal, LabelPre2, average='weighted')
        LabelPre3 = LabelPre2.copy()
        SeizureReal = 0
        SeizurePre  = 0
        Lat         = 0
        for j in range(CFG['postwindows'], len(LabelPre2)):
            if sum(LabelPre2[(j-CFG['postwindows']):j])>=CFG['postwindows']:
                LabelPre3[j]=1
            else:
                LabelPre3[j]=0
        for j in range(len(LabelReal)):
            if j==0 and LabelReal[j]==1:
                onset = j
            elif j>0 and (LabelReal[j]-LabelReal[j-1])>0:
                onset = j
            if (j>0 and (LabelReal[j]-LabelReal[j-1])<0) or (LabelReal[j]==1 and j==len(LabelReal)-1):
                offset = j
                if LabelReal[j]==1 and j==len(LabelReal)-1:
                    offset = offset + 1
                SeizureReal = SeizureReal + 1
                if sum(LabelPre3[onset:offset]==1):
                    LabelPreTemp = LabelPre3[onset:offset]
                    Lat = Lat + (list(LabelPreTemp).index(1)*CFG['poststep'])
                    SeizurePre = SeizurePre + 1
        Sen2 = SeizurePre/SeizureReal
        FDR  = sum(LabelPre3[LabelReal==0])/len(LabelPre3[LabelReal==0])*3600/CFG['poststep']
        Lat  = Lat/SeizureReal
        print('Sen1: %.4f, Spe: %.4f, Acc: %.4f, Auc: %.4f, F1-score: %.4f, Sen2: %.4f, FDR: %.2f, Lat: %.2f'
              %(Sen1, Spe, Acc, Auc, F1Binary, Sen2, FDR, Lat))
        ResultsDf = ResultsDf.append({'Sub': CFG['Subjects'][i], 'Chan': 0,
                                      'Rec': ('%.1f' % (len(LabelReal)/3600*CFG['poststep'])),
                                      'Seiz': SeizureReal, 
                                      'Sen1': Sen1, 'Spe': Spe, 'Acc': Acc, 'Auc': Auc,
                                      'F1-score': F1Binary, 'F1-score2': F1Weighted,
                                      'Sen2': Sen2, 'FDR': FDR, 'Lat': Lat,
                                      }, ignore_index=True)
    ResultsDf = ResultsDf.append({'Sub': 'Mean', 'Chan': 0,
                                  'Rec': ('%.1f' % np.mean(np.float32(ResultsDf['Rec']))),
                                  'Seiz': np.mean(ResultsDf['Seiz']), 
                                  'Sen1': np.mean(ResultsDf['Sen1']),
                                  'Spe': np.mean(ResultsDf['Spe']),
                                  'Acc': np.mean(ResultsDf['Acc']),
                                  'Auc': np.mean(ResultsDf['Auc']),
                                  'F1-score': np.mean(ResultsDf['F1-score']),
                                  'F1-score2': np.mean(ResultsDf['F1-score2']),
                                  'Sen2': np.mean(ResultsDf['Sen2']),
                                  'FDR': np.mean(ResultsDf['FDR']),
                                  'Lat': np.mean(ResultsDf['Lat']),
                                  }, ignore_index=True)
    if CFG['permutation']:
        ResultsDf.to_csv(CFG['cross_site'] + '2' + CFG['dataset'] + '_' + CFG['model'] + '_' + CFG['MinMaxMode'] +'_Permutation.csv', index=False)
    else:
        ResultsDf.to_csv(CFG['cross_site'] + '2' + CFG['dataset'] + '_' + CFG['model'] + '_' + CFG['MinMaxMode'] +'_NonPermutat.csv', index=False)
    print('\nMean')
    print('Sen1: %.4f, Spe: %.4f, Acc: %.4f, Auc: %.4f, F1-score: %.4f, Sen2: %.4f, FDR: %.2f, Lat: %.2f'
          %(np.mean(ResultsDf['Sen1']), np.mean(ResultsDf['Spe']),
            np.mean(ResultsDf['Acc']), np.mean(ResultsDf['Auc']), np.mean(ResultsDf['F1-score']),
            np.mean(ResultsDf['Sen2']), np.mean(ResultsDf['FDR']), np.mean(ResultsDf['Lat'])))


#%% main
if __name__ =='__main__':
    # 1. load info
    CFG, DatasetDf = infoloader(CFG) # 导入数据信息

    # 2. select data
    dataselecter(DatasetDf, CFG) # 数据预处理
    
    # 3. train
    ModelTrain(CFG)

    # 4. test
    ModelTest(CFG) # 读取数据和模型保存真实标签和预测标签

    # 5. postprocess
    Postprocess(CFG) # 保存测试结果模型性能


