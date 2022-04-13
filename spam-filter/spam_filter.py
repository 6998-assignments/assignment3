import json
import urllib.parse
import boto3
import io
import tempfile
import numpy as np
from sagemaker.mxnet.model import MXNetPredictor
from utils import one_hot_encode
from utils import vectorize_sequences

ENDPOINT = "sms-spam-classifier-mxnet-2022-04-12-16-08-13-155"
sage = boto3.client('runtime.sagemaker', region_name='us-east-1')
s3 = boto3.client('s3')
    
def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    print(bucket)
    print(key)
    try:
        email = s3.get_object(Bucket=bucket, Key=key)
        raw = email['Body'].read().decode('utf-8')
        print("parts")
        parts = raw.split('Content-Type: text/plain; charset="UTF-8"')
        if len(parts)>1:
            raw = parts[1].split("Content-Type: text/html;")
            parts = raw[0]
            parts = parts.split("\r\n")
            text = "".join(parts[:-2])
            print(text)
        else:
            print("unknown format")
            return ""
    except Exception as e:
        print(e)
        print("Error when get email from s3 and parse")
        raise e
    try:
        vocabulary_length = 9013
        mxnet_pred = MXNetPredictor('sms-spam-classifier-mxnet-2022-04-12-16-08-13-155')
        test_messages = [text]
        one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
        encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
        result = mxnet_pred.predict(encoded_test_messages)
        return result['predicted_label'][0]
    except Exception as e:
        print(e)
        print("Error when get pred from sagemaker")
        raise e
        
            
            