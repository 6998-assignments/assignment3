import urllib.parse
import boto3
import numpy as np
from sagemaker.mxnet.model import MXNetPredictor
from utils import one_hot_encode
from utils import vectorize_sequences

ENDPOINT = 'sms-spam-classifier-mxnet'
AWS_REGION = "us-east-1"
sage = boto3.client('runtime.sagemaker', region_name=AWS_REGION)
s3 = boto3.client('s3')

def getBody(raw_email):
    if 'Content-Type: text/plain;' in raw_email \
    and 'Content-Type: text/html;' in raw_email:
        parts = raw_email.split('Content-Type: text/plain;')
        raw = parts[1].split('Content-Type: text/html;')
        parts = raw[0]
        parts = parts.split("\n")
        text = "".join(parts[1:-2])
        if "Content-Transfer-Encoding: 7bit" in text:
            text = text.replace("Content-Transfer-Encoding: 7bit", "")
        text.replace("\n", "")
        return text
    return "no email body detected"

def getSubject(raw_email):
    if "Subject: " in raw_email:
        parts = raw_email.split("Subject: ")[1]
        subject = parts.split("\n")[0]
        return subject
    return "no email subject detected"

def getDate(raw_email):
    if "Date: " in raw_email:
        parts = raw_email.split("Date: ")[1]
        date = parts.split("\n")[0]
        return date
    return "no email date detected"

def getSender(raw_email):
    if "From: " in raw_email:
        parts = raw_email.split("From: ")[1]
        sender = parts.split("\n")[0]
        return sender
    return "no email sender detected"

def getLabel(body):
    try:
        vocabulary_length = 9013
        mxnet_pred = MXNetPredictor(ENDPOINT)
        test_messages = [body]
        one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
        encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
        result = mxnet_pred.predict(encoded_test_messages)
        label = result['predicted_label'][0][0]
        label = "Spam" if label > 0 else "Ham"
        score = result['predicted_probability'][0][0]
        score = "{:.2f}%".format(score*100)   
        return label, score
    except Exception as e:
        print(e)
        print("Error when get pred from sagemaker")
        raise e

def sendSES(sender, date, subject, body, label, score):
    SENDER = "service@nospam4ever.shop"
    RECIPIENT = sender
    SUBJECT = "Email spam filter report"
    BODY_TEXT = ("We received your email sent at {} with the subject {}.\nHere is a 240 character sample of the email body: {}\nThe email was categorized as {} with a {} confidence."
        .format(date, subject, body, label, score)) 
    CHARSET = "UTF-8"
    client = boto3.client('ses',region_name=AWS_REGION)
    try:
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {                    
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
        )
    # Display an error if something goes wrong.	
    except Exception as e:
        print(e)
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    print(bucket)
    print(key)
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        raw_email = response['Body'].read().decode('utf-8')
        body = getBody(raw_email)
        subject = getSubject(raw_email)
        sender = getSender(raw_email)
        date = getDate(raw_email)
        label, score = getLabel(body)

        print("sender: " + sender)
        print("date: " + date)
        print("subject: " + subject)
        print("body: " + body)
        print("label: " + label)
        print("score: " + score)
        sendSES(sender, date, subject, body, label, score)
    except Exception as e:
        print(e)
        print("Error when get email from s3 and parse")
        raise e