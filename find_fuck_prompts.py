import asyncio
import urllib.parse
import os

import aiohttp


base_folder = '/workspace/DB'

validation_url = urllib.parse.urljoin("http://127.0.0.1:8094", "/validation/")
validation_timeout = 14

async def validate(validation_url: str, timeout: int, prompt: str, DATA_DIR: str):
    async with aiohttp.ClientSession() as session:
        try:
            # print(f"=================================================")
            client_timeout = aiohttp.ClientTimeout(total=float(timeout))
            
            async with session.post(validation_url, timeout=client_timeout, json={"prompt": prompt, "DATA_DIR": DATA_DIR}) as response:
                if response.status == 200:
                    result = await response.json()
                    # print("Success:", result)
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
# iterate through all subfolders and files
async def main():
    count = 0
    for subdir, _, files in os.walk(base_folder):
        count += 1
        
        if count % 100 == 0:
            print(f"----------------------{count}------------------------")
        prompt_path = os.path.join(subdir, 'prompt.txt')
        img_path = os.path.join(subdir, 'img.jpg')
        mesh_path = os.path.join(subdir, 'mesh.glb')
        
        # read prompt.txt
        if os.path.isfile(prompt_path):
            with open(prompt_path, 'r') as f:
                prompt_text = f.read()
                if not "sword" in prompt_text:
                    continue
                # print(f'Text from {prompt_path}:\n{prompt_text}\n')
        
            # read img.jpg (just checking if it exists)
            if not os.path.isfile(img_path) or not os.path.isfile(mesh_path):
                print(f'Not able to proceed because there is no image file or mesh file for {prompt_text}')
            
            result = await validate(validation_url=validation_url, timeout=validation_timeout, prompt=prompt_text, DATA_DIR=subdir)
            Q0 = result['Q0']
            S0 = result['S0']
            # Logging
            if S0 < 0.23:
                with open(f"/workspace/awfulth_awful_prompts.txt", "a") as file:
                    file.write(f"{prompt_text}=={result['Q0']}={result['S0']}\n")
        
if __name__ == "__main__":
    asyncio.run(main())

