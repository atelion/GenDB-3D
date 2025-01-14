import os
import urllib.parse
import aiohttp
import asyncio
import logging
import hashlib

simplify = 0.95
texture_size = 1024
async def _generate(generation_url: str, generation_timeout: int, prompt: str, DATA_DIR: str, simplify: float, texture_size: str):
    print("Start")
    async with aiohttp.ClientSession() as session:
        try:
            print(f"==========================={DATA_DIR}======================")
            client_timeout = aiohttp.ClientTimeout(total=float(generation_timeout))
            
            async with session.post(generation_url, timeout=client_timeout, json={"prompt": prompt, "DATA_DIR": DATA_DIR}) as response:
                if response.status == 200:
                    result = await response.json()
                    logging.info("Success:", result)
                else:
                    logging.error(f"Generation failed. Please try again.: {response.status}")
                # return result
        except aiohttp.ClientConnectorError:
            logging.error(f"Failed to connect to the endpoint. Try to access again: {generation_url}.")
        except TimeoutError:
            logging.error(f"The request to the endpoint timed out: {generation_url}")
        except aiohttp.ClientError as e:
            logging.error(f"An unexpected client error occurred: {e} ({generation_url})")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e} ({generation_url})")

async def generate(prompt: str, simplify: float, texture_size: int):
    generation_url = urllib.parse.urljoin("http://127.0.0.1:8093", "/generate/")
    generation_timeout = 200
    # prompt = "ancient bronze shield with serpent motif and verdigris patina"
    simplify = 0.95
    texture_size = 1024
    DATA_DIR = "/workspace/GenDB-3D/Extra/"
    
    line_hash = hashlib.sha256(prompt.encode()).hexdigest()
    folder_path = os.path.join(DATA_DIR, line_hash)
    os.makedirs(folder_path, exist_ok=True)
    
    await _generate(generation_url, generation_timeout, prompt, folder_path, simplify, texture_size)
    logging.info("Successfully generated")

async def _validate(validation_url: str, validation_timeout: int, prompt: str, DATA_DIR: str):
    async with aiohttp.ClientSession() as session:
        try:
            print(f"=================================================")
            client_timeout = aiohttp.ClientTimeout(total=float(validation_timeout))
            
            async with session.post(validation_url, timeout=client_timeout, json={"prompt": prompt, "DATA_DIR": DATA_DIR}) as response:
                if response.status == 200:
                    result = await response.json()
                    print("Success:", result)
                    return result
                else:
                    logging.error(f"Validation failed. Please try again.: {response.status}")
                # return result
        except aiohttp.ClientConnectorError:
            logging.error(f"Failed to connect to the endpoint. Try to access again: {validation_url}.")
        except TimeoutError:
            logging.error(f"The request to the endpoint timed out: {validation_url}")
        except aiohttp.ClientError as e:
            logging.error(f"An unexpected client error occurred: {e} ({validation_url})")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e} ({validation_url})")

async def validate(prompt: str):
    validation_url = urllib.parse.urljoin("http://127.0.0.1:8094", "/validation/")
    validation_timeout = 4
    # prompt = "ancient bronze shield with serpent motif and verdigris patina"
    DATA_DIR = "/workspace/GenDB-3D/Extra"
    result = await _validate(validation_url, validation_timeout, prompt, DATA_DIR)
    print(result)
    logging.info("Successfully validated")

def main():
    prompt = "coral reef teleporter with barnacles and water damage"
    # prompt = "steampunk pocket watch with brass gears and ticking mechanisms"
    # prompt = "quantum computer terminal with hologram projector and cooling vents"
    # extra_prompts = "Angled front view, solid color background, 3d model, high quality, detailed sub components, anime style"
    # extra_prompts = "Angled front view, solid color background, 3d model, high quality, anime style"
    # extra_prompts = "Angled front view, solid color background, high quality, detailed textures, realistic lighting, emphasis on form and depth, suitable for 3D rendering."
    # extra_prompts = "anime, Angled front view, solid color background, 3d model, realistic lighting, emphasis on texture and depth, suitable for 3D rendering."
    # extra_prompts = "Angled front view, solid color background, detailed sub-components, suitable for 3D rendering, include relevant complementary objects (e.g., a stand for the clock, a decorative base for the sword) linked to the main object to create context and depth."
    
    # enhanced_prompt = f"{prompt}, {extra_prompts}"
    # Open the file in read mode
    DATA_DIR = "/workspace/GenDB-3D/Extra/"
    with open('awful_prompts.txt', 'r') as file:
        # Read and print each line
        count = 0
        for line in file:
            count += 1
            if count % 100 == 0:
                print(f"-------------------{count}---------------------\n")
            line = line.strip()
            line_hash = hashlib.sha256(line.encode()).hexdigest()
            folder_path = os.path.join(DATA_DIR, line_hash)
            os.makedirs(folder_path, exist_ok=True)
            # Define the path for the text file to save the line
            text_file_path = os.path.join(folder_path, 'prompt.txt')
            # Save the line in the text file
            with open(text_file_path, 'w') as text_file:
                text_file.write(line)
            asyncio.run(generate(prompt=line, simplify=0.95, texture_size=1024))
    # asyncio.run(validate(prompt))
if __name__ == '__main__':
    main()