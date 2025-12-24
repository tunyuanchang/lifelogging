import sys
import glob
import os
import re
import csv
import json
import pymongo
import pandas as pd

def collectShotBoundaries(shotinfodir,fps):
    videosdict = {}

    for filename in glob.iglob(shotinfodir + '*.txt', recursive=True):
        #print(filename)
        filebasename = os.path.basename(filename)
        #print(filebasename)
        videoid = filebasename.split('.')[0]
        #print(videoid)

        strshots = []
        with open(filename) as f:
            strshots = f.readlines()

        shotsdict = []
        for shot in strshots:
            tmp = shot.split(' ')
            ffrom = int(tmp[0])
            fto = int(tmp[1])

            # [TODO] select keyframes for shot
            for i in range(ffrom, fto + 1):
                fkey = i
                keyframe = f"{videoid}-{fkey}.png"
                shotobject = {
                    "from": ffrom,
                    "to": fto,
                    "keyframe": keyframe
                }
                shotsdict.append(shotobject)

            # shotsdict.append(shotobject)
            # fkey = ffrom + int((fto - ffrom)/2)
            # keyframe = f"{videoid}_{fkey}.jpg"
            # shotobject = {
            #     "from": ffrom,
            #     "to": fto,
            #     "keyframe": keyframe
            # }

            # shotsdict.append(shotobject)

        videoobject = {
            "videoid": videoid,
            "fps": fps[f'{videoid}.mp4'],
            "shots": shotsdict
        }
        videosdict[videoid] = videoobject

    return videosdict

def collectASR(speechdir, videos):
    for videoid in videos:
        filename = f"{asrdir}/{videoid}_speech_results.csv"
        if not os.path.isfile(filename):
            continue

        kfspeeches = []
        with open(filename, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader) #skip header
            for row in csv_reader:
                ffrom = int(row[0])
                fto = int(row[1])
                speech = row[2].replace(',',' ')
                speech = re.sub(r'[^\w\s]', '', speech)  # Keeps only alphanumeric characters and spaces
                speech = re.sub(r'\s+', ' ', speech)   # Replace multiple spaces/newlines with a single space
                speech = speech.lower().strip()
                if len(speech) < 2:
                    continue

                for shot in videos[videoid]['shots']:
                    if ffrom >= shot['from'] and ffrom <= shot['to'] or fto >= shot['from'] and fto <= shot['to']:
                        obj = {
                            "speech": speech,
                            "keyframe": shot['keyframe']
                        }
                        kfspeeches.append(obj)
                        break
                
        videos[videoid]['asr'] = kfspeeches

def collectOCR(textsdir, videos):
    for videoid in videos:
        filename = f"{ocrdir}/ocr_{videoid}.csv"
        if not os.path.isfile(filename):
            continue
        
        kftexts = []

        with open(filename, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)  # Default delimiter is ','
            next(csv_reader) #skip header
            for row in csv_reader:
                kf = row[0]  # First column
                alltexts = row[1]  # Second column
                texts = alltexts.split(',')

                for text in texts:
                    # Replace all special characters with an empty string
                    text = re.sub(r'[^\w\s]', '', text)  # Keeps only alphanumeric characters and spaces
                    text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces/newlines with a single space
                    text = text.lower().strip()
                    if len(text) < 2:
                        continue

                    comps = kf.split('_')
                    sec = float(comps[len(comps)-1].split('.')[0]) / fps[f"{videoid}.mp4"]
                    obj = {
                        "text": text,
                        "second": sec,
                        "keyframe": kf
                    }
                    kftexts.append(obj)

        videos[videoid]['texts'] = kftexts




myclient = pymongo.MongoClient('mongodb://localhost:27017')    
mydb = myclient['divexplore']

fpsfilename = sys.argv[1]
shotinfodir = sys.argv[2]
ocrdir = sys.argv[3]
asrdir = sys.argv[4]

print('reading fps')
df = pd.read_csv(fpsfilename, sep=' ', header=None, index_col=0).squeeze("columns")
fps = df.to_dict()

print('reading shot boundaries')
videosdict = collectShotBoundaries(shotinfodir,fps)

print('collecting ocr data')
collectOCR(ocrdir, videosdict)

print('collecting asr data')
collectASR(asrdir, videosdict)

print(f'connecting to MongoDB at {myclient}')

mytxt = mydb['texts']
mytxt.drop()

mycol = mydb['videos']
mycol.drop()
mycol.insert_many(videosdict.values())

# print(videosdict)