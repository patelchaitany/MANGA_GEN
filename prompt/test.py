import requests

image_url = "https://otakukart.com/wp-content/uploads/2023/05/power-2.jpg"

response = requests.get(image_url)
print(response.status_code)  # 200 means it's accessible
print(response.headers)  # Check for content type and restrictions
