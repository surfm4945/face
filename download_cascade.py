import urllib.request

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
filename = "haarcascade_frontalface_default.xml"

print("Downloading Haar Cascade file...")
urllib.request.urlretrieve(url, filename)
print("Download complete! File saved as:", filename)