from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
import pandas as pd
import joblib
pipe_lr = joblib.load(open("textapp/emotion_classifier_pipe_lr.pkl", "rb"))
# textapp/views.py
@csrf_exempt
def text_analysis(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_text = data.get('user_text', '')
        # Process the text as needed
        string=str(pipe_lr.predict([user_text]))
        string=string[2:]
        string=string[:-2]
        print(f"This is the {string} in terminal")
        return JsonResponse({'message': 'Text received', 'text': string})
    return render(request, 'text_analysis.html')
# Create your views here.
