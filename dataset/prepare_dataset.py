import json,os, requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import random


def process_csv_files(categories):
    data = pd.concat([pd.read_csv(file) for file in ['processed_data_1.csv', 'processed_data_2.csv', 'processed_data_2.csv']])
    if categories:
        data = data[data['category'].isin(categories)]
        processed_dict = {}
        for i in data.iloc:
            if i['id'] not in processed_dict:
                processed_dict[i['id']]=i['color']        
        data = data[[ True if processed_dict[i['id']]== i['color'] else False for i in data.iloc]]
    return data

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
    with ThreadPoolExecutor(max_workers=300) as executor:
        future_to_image = {executor.submit(download_image, *task): task for task in image_tasks}

        for future in as_completed(future_to_image):
            task = future_to_image[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error in task {task}: {e}")

def main(categories, train_images,test_images, dataset_folder='fashion_images'):
    try:
        data = process_csv_files(categories.split(','))
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
    parser.add_argument('-c', '--categories', default='minidress,skirt,gown,blouse,shorts,sweater,pants,jacket,jeans,top,dress,tee')      # option that takes a value
    parser.add_argument('-train-images', '--train-images', type=int, default= None, help='number of images to download for training dataset')
    parser.add_argument('-test-images', '--test-images', type=int, default= None, help='number of images to download for training dataset')
    parser.add_argument('-dataset_folder', '--dataset-folder', type=str, default= "fashion_images", help="path to download images")
    args = parser.parse_args()
    main(args.config_file,args.train_images,args.test_images,args.dataset_folder)


