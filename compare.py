import os
import shutil
import aiohttp
import time
from PIL import Image
from fastapi import FastAPI, HTTPException, Body
import hashlib
import urllib.parse
import asyncio

async def validate(validation_url: str, timeout: int, prompt: str, DATA_DIR: str):
    async with aiohttp.ClientSession() as session:
        try:
            print(f"=================================================")
            client_timeout = aiohttp.ClientTimeout(total=float(timeout))
            
            async with session.post(validation_url, timeout=client_timeout, json={"prompt": prompt, "DATA_DIR": DATA_DIR}) as response:
                if response.status == 200:
                    result = await response.json()
                    print("Success:", result)
                else:
                    print(f"Validation failed. Please try again.: {response.status}")
                return result
        except aiohttp.ClientConnectorError:
            print(f"Failed to connect to the endpoint. Try to access again: {validation_url}.")
        except TimeoutError:
            print(f"The request to the endpoint timed out: {validation_url}")
        except aiohttp.ClientError as e:
            print(f"An unexpected client error occurred: {e} ({validation_url})")
        except Exception as e:
            print(f"An unexpected error occurred: {e} ({validation_url})")
    
    return None

async def get_scores(prompt, old_path: str, new_path: str):
    validation_url = urllib.parse.urljoin("http://127.0.0.1:8094", "/validation/")
    validation_timeout = 50
    try:
        result_old = await validate(validation_url=validation_url, timeout=validation_timeout, prompt=prompt, DATA_DIR = old_path)
    except:
        print("Failed in validation for old, hehehe")
    
    try:
        result_new = await validate(validation_url=validation_url, timeout=validation_timeout, prompt=prompt, DATA_DIR = new_path)
    except:
        print("Failed in validation for new, hehehe")
    
    print(f"Old result: {result_old}\nNew result: {result_new}\n")
    with open("/workspace/compare_result.txt", "a") as history:
        history.write(f"{prompt} : {result_old} : {result_new}\n")
    if result_old > result_new:
        shutil.rmtree(new_path)


def main():
    input_file = '/workspace/update_db/warning.txt'
    new_directory = '/workspace/warndb'
    old_directory = '/workspace/DB'
    
    inputfile = open(input_file, "r")
    lines = inputfile.readlines()
    
    for id, line in enumerate(lines):
        id += 1
        if id % 10 == 0:
            print(id)
        # Remove any leading/trailing whitespace
        line = line.strip()
        print(line)
        if line:  # Ensure the line is not empty
            # Create a hash of the line
            line_hash = hashlib.sha256(line.encode()).hexdigest()
            # Create a folder with the hash name
            old_path = os.path.join(old_directory, line_hash)
            new_path = os.path.join(new_directory, line_hash)
            
            prompt = line
            asyncio.run(get_scores(prompt, old_path, new_path))
    inputfile.close()

if __name__ == "__main__":
    main()
        
    
    




