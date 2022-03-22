from django.test import TestCase
import os

# Create your tests here.
dir_path = os.path.join(os.path.join(os.getcwd(),"static") , 'recordings')

# absolutepath = os.path.abspath(__file__)
# fileDirectory = os.path.dirname(absolutepath)
# parentDirectory = os.path.dirname(fileDirectory)
# dir_path = os.path.join(parentDirectory , 'static\\recordings')

print(dir_path)
# C:\Users\saleh\Desktop\django\blog\test\project\static\recordings
