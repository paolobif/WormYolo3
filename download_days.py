from download import *

API_ENDPOINT = 'http://wormbotuser:wormbotpassword@128.95.23.15/wormbot/'

def download_video_day(video_name: str, save: str, endpoint: str = API_ENDPOINT):

  for i in range(3,21):
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
      print(f"{video_name}")
    
  return True
  # print(f"Downlaoded experiment {video_name}")
def download_video_day_list(videos: list, save: str, endpoint: str = API_ENDPOINT):
    """Downloads list of day videos from Wormbot api.

    Args:
        videos (list): list of videos.
        save (str): save path.
        endpoint (str, optional): endpoint url. Defaults to API_ENDPOINT.
    """
    success = []
    fail = []
    for video in videos:
        status = download_video_day(video, save, endpoint)
        if status:
            success.append(video)
        if not status:
            fail.append(video)

    print(f"Failed to download: {fail}")
    print(f"Successfuly downloaded: {success}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-vids', type=str,
                        help="""String of videos separated by commas or use
                                ':' to specify range. ie '2:6, 3, 4, 7:10'
                                Remeber - last number is not inclusive.""")
    parser.add_argument('-save', type=pathlib.Path, default="videos",
                        help="""Path to download directory""")

    args = parser.parse_args()

    if not args.vids:
        print("""
                 Select videos you wish to download \n
                 Use ':' to indicate range of values (not inclusive) and ','
                 to separate values.\n ie '2:6, 3, 4, 7:10'
              """)
        input_string = input("Enter here: ")
    else:
        input_string = args.vids
        print(f"Args provided: {input_string}")

    videos = parse_vids_arg(input_string)  # Inputed video list.
    save_path = args.save  # Where downloaded videos are to be saved.

    # Run download.
    download_video_day_list(videos, save_path, API_ENDPOINT)


