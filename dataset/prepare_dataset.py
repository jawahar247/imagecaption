import json,os, requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import random



def process_image_caption_json_file(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
        image_data_dict = {"id":[],"image_url":[],"title":[],"description":[],"categoryid":[],"category":[],"color":[], 'image_path': []}
        for num, d in enumerate(data):
            for image in d['images']:
                for key, i in image.items():
                    if key == 'color':
                        continue
                    image_data_dict['id'].append(d['id'])
                    image_data_dict['title'].append(d['title'])
                    image_data_dict['description'].append(d['description'])
                    image_data_dict['categoryid'].append(d['categoryid'])
                    image_data_dict['category'].append(d['category'])
                    if not i:
                        image_data_dict['image_url'].append(i)
                        image_data_dict['image_path'].append(f'{key}.jpeg')
                    else:
                        image_data_dict['image_url'].append(i.split('?', 1)[0])
                        image_data_dict['image_path'].append(f'{key}.jpeg')
                    image_data_dict['color'].append(image['color'])
        data_dict = pd.DataFrame(image_data_dict)
        data_dict.to_csv("processed_full_data.csv",index=False)


def download_image(image_url, file_path):
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        print(f"Image downloaded and saved as {file_path}")

    except requests.RequestException as e:
        print(f"Error downloading the image: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def download_images_in_parallel(image_tasks):
    with ThreadPoolExecutor(max_workers=100) as executor:
        future_to_image = {executor.submit(download_image, *task): task for task in image_tasks}

        for future in as_completed(future_to_image):
            task = future_to_image[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error in task {task}: {e}")

def main(config_file, train_images,test_images, dataset_folder='fashion_images'):
    try:
        if config_file.endswith('.json'):
            process_image_caption_json_file(config_file)
        if config_file.endswith('.csv'):
            data = pd.read_csv(config_file)
        else:
            data = pd.read_csv("processed_full_data.csv")
            data = data.sample(frac=1)

        if train_images and test_images:
            data = data[0: train_images+test_images+10]
        
        image_tasks = []
        images = {'image':[], 'description': []}
        for count, items in enumerate(data.iloc):
            path = os.path.join(dataset_folder, items['category'])
            if not os.path.exists(path):
                os.makedirs(path)
            path = f"{path}/{items['id']}_{items['color'].replace('/', '_')}_{items['image_path']}".replace(' ', '')
            if not os.path.exists(path):
                image_tasks.append((items["image_url"], path))
            images['image'].append(path)
            images['description'].append(items['description'])
            
            if train_images and test_images:
                if count >= train_images+test_images:
                    pd.DataFrame(images).to_csv(f'test_images_{test_images}.csv', index=False)
                    break
                elif count >= train_images:
                    pd.DataFrame(images).to_csv(f'train_images_{train_images}.csv', index=False)
                    images = {'image':[], 'description': []}
                    
        if not train_images and not test_images:
            pd.DataFrame(images).to_csv(f'full_images.csv', index=False)

        print(f"Number of images to download: =  {len(image_tasks)}")
        download_images_in_parallel(image_tasks)
        print(f"length of image task {len(image_tasks)}")

    except FileNotFoundError as e:
        print(f"Error: The file was not found: {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
                    prog='fashon_image_data processing',
                    description='input meta_all_129927.json download required images')
    parser.add_argument('-c', '--config_file', default='meta_all_129927.json')      # option that takes a value
    parser.add_argument('-train-images', '--train-images', type=int, default= None, help='number of images to download for training dataset')
    parser.add_argument('-test-images', '--test-images', type=int, default= None, help='number of images to download for training dataset')
    parser.add_argument('-dataset_folder', '--dataset-folder', type=str, default= "fashion_images", help="path to download images")
    args = parser.parse_args()
    main(args.config_file,args.train_images,args.test_images,args.dataset_folder)


