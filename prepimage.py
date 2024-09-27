import base64

# Load the image and encode it as base64
with open('./test3.jpeg', 'rb') as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Save the base64 string to a text file
with open('encoded_image3.txt', 'w') as text_file:
    text_file.write(encoded_image)

print("Base64 encoded image saved to encoded_image.txt")