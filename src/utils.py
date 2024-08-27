from src.log import logging
from src.exception import CustomException
from huggingface_hub._login import notebook_login

def hugging_face_login():
    notebook_login()


def open_file():

    with open("file_name",'f') as file:
        data = file.readlines()
    return data




