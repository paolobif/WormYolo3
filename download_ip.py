import os
import argparse
import pathlib
import requests
import urllib.parse
import sys

"""
    This script downloads videos from the Wormbot server that are specified
    when the script is run. Saves in a ./results by defualt but can be
    changed by specifying the save path using -save.
    example usage:
        python3 download.py -vids "660:665, 550, 556, 300:355"
        or
        python3 download.py
            and then specify videos when prompted.
"""
API_BASE = "http://wormbotuser:wormbotpassword@"
API_END = "/wormbot/"
api_endpoint = 'http://wormbotuser:wormbotpassword@128.95.23.14/wormbot/'


def parse_vids_arg(input_string: str):
    """Parses input string when script is called.

    Args:
        input_string ([str]): [Notation for videos to be downloaded]

    Returns:
        [list]: [list of videos to download]
    """
    input_string = input_string.replace(" ", "")  # Remove empty spaces.
    vals = input_string.split(',')
    videos = []
    for n in vals:
        if ":" in n:
            a, b = n.split(":")
            vid_range = [i for i in range(int(a), int(b))]
            videos.extend(vid_range)
        else:
            videos.extend([int(n)])

    return videos


def download_video(video_name: int, save: str, endpoint: str):
    video_endpoint = f"{video_name}/{video_name}.avi"
    url = urllib.parse.urljoin(endpoint, video_endpoint)
    log_file = open(os.path.join(cur_dir,"download_log.txt"),"a")
    log_file.write(url+"\n")
    log_file.close()
    video_file = requests.get(url)

    if video_file.status_code != 200:
        log_file = open(os.path.join(cur_dir,"download_log.txt"),"a")
        log_file.write(f"error downloading experiment {video_name}\n")
        log_file.close()
        return(False)

    if not os.path.exists(save):
        print("making save path")
        os.makedirs(save)

    file_name = os.path.join(save, str(video_name) + ".avi")
    with open(file_name, "wb") as file:
        file.write(video_file.content)

    #file_name = os.path.join(save, str(video_name) + "_description.txt")
    #with open(file_name, "wb") as file:
    #    file.write(video_file.content)


    # print(f"Downlaoded experiment {video_name}")
    return(True)

def download_video_list(videos: list, save: str, endpoint: str = api_endpoint):
    """Downloads list of videos form Wormbot api.

    Args:
        videos (list): list of videos.
        save (str): save path.
        endpoint (str, optional): endpoint url. Defaults to api_endpoint.
    """
    success = []
    fail = []
    for video in videos:
        status = download_video(video, save, endpoint)
        if status:
            success.append(video)
        if not status:
            fail.append(video)
    log_file = open(os.path.join(cur_dir,"download_log.txt"),"a")
    log_file.write(f"Failed to download: {fail}\n")
    log_file.write(f"Successfuly downloaded: {success}\n")
    log_file.close()

def download_video_day(video_name: str, save: str, endpoint: str = api_endpoint, first_day = 3, last_day = 3):

  for i in range(first_day,last_day+1):
    video_endpoint = f"{video_name}/day{i}.avi"

    url = urllib.parse.urljoin(endpoint, video_endpoint)
    print(url)
    video_file = requests.get(url)

    if video_file.status_code != 200:
      print(f"error downloading experiment {video_name}")

    """
    if not os.path.exists(str(save)+"/"+str(video_name)):
      print("making save path")
      os.makedirs(str(save)+"/"+str(video_name))
    """

    file_name = os.path.join(save, str(video_name) +"_day"+str(i) + ".avi")
    with open(file_name, "wb") as file:
      file.write(video_file.content)
      log_file.write(f"{video_name}\n")

  return True
  # print(f"Downlaoded experiment {video_name}")
def download_video_day_list(videos: list, save: str, endpoint: str = api_endpoint, first_day = 3, last_day = 20):
    """Downloads list of day videos from Wormbot api.

    Args:
        videos (list): list of videos.
        save (str): save path.
        endpoint (str, optional): endpoint url. Defaults to api_endpoint.
    """
    success = []
    fail = []
    for video in videos:
        status = download_video_day(video, save, endpoint, first_day, last_day)
        if status:
            success.append(video)
        if not status:
            fail.append(video)

    log_file = open(os.path.join(cur_dir,"download_log.txt"),"a")
    log_file.write(f"Failed to download: {fail}\n")
    log_file.write(f"Successfuly downloaded: {success}\n")
    log_file.close()

global log_file
log_file = None

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-vids', type=str,
                        help=\"""String of videos separated by commas or use
                                ':' to specify range. ie '2:6, 3, 4, 7:10'
                                Remeber - last number is not inclusive.\""")
    parser.add_argument('-save', type=pathlib.Path, default="videos",
                        help=\"""Path to download directory\""")


    args = parser.parse_args()

    if not args.vids:
        print(\"""
                 Select videos you wish to download \n
                 Use ':' to indicate range of values (not inclusive) and ','
                 to separate values.\n ie '2:6, 3, 4, 7:10'
              \""")
        input_string = input("Enter here: ")

    else:
        input_string = args.vids
        print(f"Args provided: {input_string}")
    """

    file_path = os.path.abspath(__file__)
    cur_dir = os.path.split(file_path)[0]

    log_file = open(os.path.join(cur_dir,"download_log.txt"),"w+")
    log_file.write(str(os.getpid())+"\n")
    log_file.close()

    api_endpoint = API_BASE + sys.argv[1] + API_END

    input_string = sys.argv[2]

    videos = parse_vids_arg(input_string)  # Inputed video list.
    save_path = sys.argv[3]  # Where downloaded videos are to be saved.

    download_days = sys.argv[4] == "True"

    if download_days:
      first_day = int(sys.argv[5])
      last_day = int(sys.argv[6])
      download_video_day_list(videos, save_path, api_endpoint, first_day, last_day)

    else:
      # Run download.
      download_video_list(videos, save_path, api_endpoint)
