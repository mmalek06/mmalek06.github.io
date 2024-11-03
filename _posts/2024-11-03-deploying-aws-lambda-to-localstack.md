---
layout: post
title: "Deploying AWS lambda to LocalStack with Terraform"
date: 2024-11-03 00:00:00 -0000
categories: Cloud
tags: ["cloud", "aws", "terraform", "localstack", "python", "sklearn", "sam-cli"]
---

# Deploying AWS lambda to LocalStack with Terraform

With my R-CNN model training, I suddenly have a lot of free time, so I decided to use it to learn something new—specifically, LocalStack and Terraform integration. In this post, I'll create an AWS Lambda that wraps an sklearn `LogisticRegression` predictor (after all, this blog focuses on ML, AI, and cloud). I'll deploy it locally, first using the `sam cli` AWS tool, then to a LocalStack container, and finally to the cloud to confirm that everything works consistently across environments.

I could have focused on just one part - the one mentioned in the title - but as a fullstack developer, I wanted to bundle everything together. This blog mainly serves as a personal knowledge repository, and I remember best when I have the full context.

## The code - part 1: training LogisticRegression model

Since the code is very short, I'll just dump it all into one snippet:

```python
import joblib
import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

with open("neg_tweets.txt", "r", encoding="utf-8") as f:
    neg_tweets = f.readlines()
    
with open("pos_tweets.txt", "r", encoding="utf-8") as f:
    pos_tweets = f.readlines()

neg_tweets = [tweet.strip() for tweet in neg_tweets]
pos_tweets = [tweet.strip() for tweet in pos_tweets]
neg_df = pd.DataFrame(neg_tweets, columns=["post"])
neg_df["sentiment"] = 0
pos_df = pd.DataFrame(pos_tweets, columns=["post"])
pos_df["sentiment"] = 1
tweets_df = pd.concat([neg_df, pos_df], ignore_index=True)
tweets_df = shuffle(tweets_df, random_state=42).reset_index(drop=True)
X_train, X_test, y_train, y_test = train_test_split(tweets_df["post"], tweets_df["sentiment"], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression()

model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("Classification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, os.path.join("..", "saved_models", "logistic_regression_model.pkl"))
joblib.dump(vectorizer, os.path.join("..", "saved_models", "vectorizer.pkl"))
```

This is almost a bootcamp-level example of training a `LogisticRegression` model. I wanted this post to tie into ML, so that's the approach I took. I'm saving both the model and the vectorizer, as both will be needed for the Lambda function to work. The classification report shows that the model isn't particularly strong in identifying most of the positive posts, but that's understandable - the positive sample size was only half that of the negative sample. Here's the report:

<pre>
Classification Report:
               precision    recall  f1-score   support

           0       0.80      0.99      0.88       282
           1       0.92      0.40      0.56       119

    accuracy                           0.81       401
   macro avg       0.86      0.69      0.72       401
weighted avg       0.83      0.81      0.79       401
</pre>

As you can see, the model is able to identify and correctly classify only 40% of the total positive examples, but let's not be bothered by that, as that's just a hello-world kind of example - it just needs to output anything.

## The code - part 2: defining lambda function and sam templates

To create the Lambda folder, I used `sam cli` and added Lambda layers on top of it. You'll often encounter Lambda layers in projects with a sufficiently large number of Lambdas, because at some point, as in any big project, code tends to repeat. Lambda layers allow you to place shared code or libraries inside them (especially useful if, for some reason, you don't want to publish them in your company's artifact repository).

Here, I'm using Lambda layers in another common way - by placing my machine learning models there, so the Lambda function can access them. Below is the project layout:

<img style="display: block; margin: 0 auto; margin-top: 15px;" src="https://mmalek06.github.io/images/lambda_terraform_project_layout.png" /><br />

And the lambda code:

```python
import os

import joblib
import json

from aws_lambda_powertools.utilities.data_classes import event_source, APIGatewayProxyEvent
from aws_lambda_powertools.utilities.typing import LambdaContext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

if os.getenv("AWS_SAM_LOCAL"):
    model_file = os.path.join("app", "model_layer", "python", "model", "logistic_regression_model.pkl")
    vectorizer_file = os.path.join("app", "model_layer", "python", "model", "vectorizer.pkl")
else:
    model_file = os.getenv("MODEL_FILE")
    vectorizer_file = os.getenv("VECTORIZER_FILE")

model: LogisticRegression = joblib.load(model_file)
vectorizer: TfidfVectorizer = joblib.load(vectorizer_file)


@event_source(data_class=APIGatewayProxyEvent)
def lambda_handler(event: APIGatewayProxyEvent, context: LambdaContext) -> dict:
    input = event.json_body["input"]
    input_vector = vectorizer.transform([input])
    prediction = int(model.predict(input_vector)[0])

    return {
        'statusCode': 200,
        'body': json.dumps(
            {
                "predicted_label": prediction,
            }
        )
    }
```

The `AWS_SAM_LOCAL` env variable is set by sam itself when you run `sam local start-api`. As for the logic - it isn't very complex, but I want to highlight one key point: the use of the `aws-lambda-powertools` library. Coming from a background in compiled languages (C#, Scala), I often find it frustrating when Python programmers skip type hints, relying on dictionaries and magic strings instead. We can make our lives much easier by leveraging all the available tools in our chosen language.

In the Python Lambda context, that's where the `aws-lambda-powertools` library shines. It provides type-safe members specific to certain Lambda trigger types. Here, since the Lambda will be called via an API Gateway, we use the appropriate decorator, specifying that we expect an `APIGatewayProxyEvent`. This allows me to use the exposed `json_body` property directly within the function. And they really have it all. Is your lambda processing Kafka events? Use `@event_source(data_class=KafkaEvent)`. Is it using Kinesis Event Streams? Use: `@event_source(data_class=KinesisStreamEvent)`.

<b>Side note:</b> If you find yourself working with AWS services using the `boto3` library, there are accompanying type-hint libraries for various IDEs: `mypy-boto3-*`, where the asterisk represents the specific AWS service name, such as `iam`, `s3`, etc.

Now to the sam template:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  SentimentPredictorFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: app.app.lambda_handler
      Runtime: python3.12
      CodeUri: ./
      MemorySize: 128
      Timeout: 900
      Environment:
        Variables:
          MODEL_FILE: "/opt/python/model/logistic_regression_model.pkl"
          VECTORIZER_FILE: "/opt/python/model/vectorizer.pkl"
      Layers:
        - !Ref ModelLayer
      Events:
        SentimentPostAPI:
          Type: Api
          Properties:
            RestApiId: !Ref SentimentPredictorApi
            Path: /predict
            Method: POST

  SentimentPredictorApi:
    Type: AWS::Serverless::Api
    Properties:
      Name: SentimentPredictorApi
      StageName: prod

  ModelLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: ModelLayer
      ContentUri: app/model_layer/
      CompatibleRuntimes:
        - python3.12

Outputs:
  SentimentPredictorApiUrl:
    Value: !Sub "https://${SentimentPredictorApi}.execute-api.${AWS::Region}.amazonaws.com/prod/predict"
    Description: "API endpoint URL for sentiment prediction"
```

The sam-cli generated template had a lot of noise in it, so I decided to rewrite it from scratch. The environment variables will be used by the Lambda file when it's deployed to the cloud or LocalStack. The paths are not random - it turns out that the assets found in Lambda Layers get deployed to the `/opt` location. Another important thing to notice is the `RestApiId` in the `Events` section of `SentimentPredictorFunction`. What it does is it connects the Lambda with the api that will be provisioned when we run `sam deploy` command - without it AWS would say that no URL was found in the api.

This was fun, especially so, because everything is working correctly. If I run this lambda locally using sam cli, it's reachable under `http://127.0.0.1:3000/predict` and if I deploy it to AWS, it also gets exposed publicly: `https://random-string.execute-api.us-east-1.amazonaws.com/prod/predict`. Now onto more fun and that's deploying it all to LocalStack using Terraform!

## The code - part 3: creating terraform definitions and deploying to LocalStack

IN PROGRESS

## Summary

IN PROGRESS
