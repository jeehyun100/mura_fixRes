import os
import requests

def download_modelfile(model_path, data_file):
    """
    Download file from <url>

    :param url: URL to file
    :param file: Local file path
    """

    model_path_and_file = model_path+"/"+ data_file
    if not os.path.isfile(model_path_and_file):
        print('Downloading ' + model_path_and_file + '...')
        r = requests.get("http://pattern:term123!@yewoo.synology.me:5005/ftp_data/mura/mura_ds.tar.gz")
        if r.status_code == 200:
            #img = r.raw.read()
            with open(model_path_and_file, 'wb') as f:
                f.write(r.content)

if __name__ == "__main__":
    # preprocess_dir = "./MURA-v1.1/"
    # desease_type = "XR_ELBOW"
    data_file = 'mura_ds.tar.gz'
    data_path = '../MURA_temp'
    os.makedirs(data_path, exist_ok=True)
    download_modelfile(data_path, data_file)
