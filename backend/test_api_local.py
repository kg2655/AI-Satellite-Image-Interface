import asyncio
from main import detect_change
from fastapi import UploadFile
import io
import traceback
import sys

async def test():
    try:
        with open('../dataset/LEVIR_CD/test/A/train_638.png', 'rb') as f1:
            with open('../dataset/LEVIR_CD/test/A/train_638.png', 'rb') as f2:
                # Fast API UploadFile mock
                class DummyUploadFile:
                    def __init__(self, f):
                        self.file = f
                
                u1 = DummyUploadFile(f1)
                u2 = DummyUploadFile(f2)
                
                result = await detect_change(u1, u2)
                print("SUCCESS!", type(result))
    except Exception as e:
        traceback.print_exc()

asyncio.run(test())
