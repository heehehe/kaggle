import os
import re
import json
import glob
from collections import defaultdict
import pandas as pd
import numpy as np


def clean_text(txt):
    ## 대회 내 주어지는 text 전처리 함수 (최종 결과물에 해당 함수 적용해야 함)
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()


def read_append_return(filename, train_files_path=train_files_path, output='text'):
    ## json 파일 읽어서 text 데이터를 dataframe에 append
    json_path = os.path.join(train_files_path, (filename+'.json'))
    headings = []
    contents = []
    combined = []
    with open(json_path, 'r') as f:
        json_decode = json.load(f)
        for data in json_decode:
            headings.append(data.get('section_title'))
            contents.append(data.get('text'))
            combined.append(data.get('section_title'))
            combined.append(data.get('text'))
    
    all_headings = ' '.join(headings)
    all_contents = ' '.join(contents)
    all_data = '. '.join(combined)
    
    if output == 'text':
        return all_contents
    elif output == 'head':
        return all_headings
    else:
        return all_data


def text_cleaning(text):
    ## text 전처리 (lowercase화, 특정 char 제거, lemmatize)
    text = ''.join([k for k in text if k not in string.punctuation])
    text = re.sub('r[^\w\s]', ' ', str(text).lower()).strip()
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    text = lem.lemmatize(text)
    return text


def predict():
    ## baseline 예측
    temp_1 = [x.lower() for x in train_df['dataset_label'].unique()]
    temp_2 = [x.lower() for x in train_df['dataset_title'].unique()]
    temp_3 = [x.lower() for x in train_df['cleaned_label'].unique()]

    existing_labels = set(temp_1 + temp_2 + temp_3)
    id_list = []
    lables_list = []
    for index, row in tqdm(sample_sub.iterrows()):
        sample_text = row['text']
        row_id = row['Id']
        temp_df = train_df[train_df['text'] == text_cleaning(sample_text)]
        cleaned_labels = temp_df['cleaned_label'].to_list()
        for known_label in existing_labels:
            if known_label in sample_text.lower():
                cleaned_labels.append(clean_text(known_label))
        for allword in set(allwords):
            if allword in sample_text.lower():
                cleaned_labels.append(clean_text(allword))
        cleaned_labels = [clean_text(x) for x in cleaned_labels]
        cleaned_labels = set(cleaned_labels)
        lables_list.append('|'.join(cleaned_labels))
        id_list.append(row_id)


def main():
    ## read data
    train_df   = pd.read_csv('../input/coleridgeinitiative-show-us-the-data/train.csv')
    sample_sub = pd.read_csv('../input/coleridgeinitiative-show-us-the-data/sample_submission.csv')
    train_files_path = '../input/coleridgeinitiative-show-us-the-data/train'
    test_files_path  = '../input/coleridgeinitiative-show-us-the-data/test'
    
    ## read_append_return
    train_df['text']   = train_df['Id'].progress_apply(read_append_return)
    sample_sub['text'] = sample_sub['Id'].progress_apply(partial(read_append_return, train_files_path=test_files_path))
    
    ## text_cleaning
    train_df['text']   = train_df['text'].progress_apply(text_cleaning)
    sample_sub['text'] = sample_sub['text'].progress_apply(text_cleaning)
    
    ## submission
    submission = pd.DataFrame()
    submission['Id'] = id_list
    submission['PredictionString'] = lables_list
    submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
