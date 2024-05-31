# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import math

# Browser
import requests

# Working with images
from PIL import Image as PILImage
from io import BytesIO

# -------------------------------------------------------------------------- #
# ---------------------------- ChatGPT Interface --------------------------- #
        
class imageModifications:
    
    def __init__(self, saveImageFolder):
        self.saveImageFolder = saveImageFolder     
        os.makedirs(self.saveImageFolder, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # ------------------------- Image Modifications ------------------------ #
    
    # Define the function to make the top half of an image translucent
    def make_top_half_translucent(self, imageRGBA):
        # Initialize the image variables.
        width, height = imageRGBA.size  # Get the size of the image
        pixels = imageRGBA.load()       # Load the pixels of the image into a variable
    
        for y in range(height // 2):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                pixels[x, y] = (r, g, b, 0)  # Set alpha to 0
    
        return imageRGBA
    
    def remove_hex_color(self, imageRGBA, hex_color):
        # Convert the hex color to a tuple of RGB
        hex_color = hex_color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        # Process the pixels
        data = imageRGBA.getdata()
        new_data = []
        for item in data:
            # Change all occurrences of the specified color to transparent
            if item[0:3] == rgb_color:
                new_data.append((255, 255, 255, 0))  # Change the color to white and set alpha to 0
            else:
                new_data.append(item)
                
        # Update image data
        imageRGBA.putdata(new_data)
        
        return imageRGBA
    
    
    def remove_similar_colors(self, imageRGBA, hex_color, tolerance=50):
        def calculate_distance(color1, color2):
            # Euclidean distance
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

        # Convert the hex color to a tuple of RGB
        hex_color = hex_color.lstrip('#')
        target_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # Process the pixels
        data = imageRGBA.getdata()
        new_data = []
        for item in data:
            color_distance = calculate_distance(item[0:3], target_color)

            # Change colors within the tolerance to transparent
            if color_distance < tolerance:
                new_data.append((255, 255, 255, 0))  # Change the color to white and set alpha to 0
            else:
                new_data.append(item)

        # Update image data
        imageRGBA.putdata(new_data)

        return imageRGBA
    
    # ---------------------------------------------------------------------- #
    # ------------------------ Modify Image Formats ------------------------ #
    
    def pullDownWebImage(self, image_url):
        # Grab the image from the URL.
        webImage = requests.get(image_url)

        # Check if the request was successful (status code 200)
        if webImage.status_code == 200:
            # Open the image and convert it to RGBA
            imageRGBA = PILImage.open(BytesIO(webImage.content)).convert('RGBA')
        
            return imageRGBA
        else:
            assert False, f"Failed to retrieve the image. Status code: {webImage.status_code}"
    
    def rbga2ByteArray(self, imageRGBA):
        # Save the image to a BytesIO object and reset the pointer
        imageByteArray = BytesIO()
        imageRGBA.save(imageByteArray, format='PNG')
        imageByteArray.seek(0)
        
        return imageByteArray
        
    # ---------------------------------------------------------------------- #
    # ------------------------ Read and Save Images ------------------------ #
    
    def saveImageURL(self, image_url, imageFilename):
        # Pull down the image from the browser.
        imageRGBA = self.pullDownWebImage(image_url)
        
        # Save the image in the saveImageFolder.
        self.saveImage(imageRGBA, imageFilename)
        
    def saveImage(self, imageRGBA, imageFilename):
        # Save the image as a PNG file
        imageRGBA.save(self.saveImageFolder + imageFilename)
        
    def readImage(self, imageFilename):
        # Open the converted image in binary read mode
        return open(self.saveImageFolder + imageFilename, "rb")
 
    # ---------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Instantiate class.
    imageController = imageModifications()
    








