from django.shortcuts import render
import os

# Create your views here.

def index(request):
    files = list()
    # rec_path = "./../project/static/recordings"
    rec_path  = os.path.join(os.path.join(os.getcwd(),"static") , 'recordings')
    recs =os.listdir(rec_path)
    # print(recs)
    for rec in recs:
      record = {rec:os.path.join(rec_path,rec)}
      files.append(record)
    # print(files)
    return render(request, 'pages/index.html',{"files":files})
    

