import os
import requests
import base64
from PIL import Image
import io

# --- Configuration ---
# The API endpoint for the image generation model.
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict"
# IMPORTANT: You need to provide your own API key.
# The environment should provide this, but if not, replace the empty string.
API_KEY = "" 
# Folder where the generated images will be saved.
OUTPUT_FOLDER = "generated_digits"

def generate_digit_images(digit: int, num_images: int = 5):
    """
    Generates and saves a specified number of handwritten digit images.

    Args:
        digit (int): The digit (0-9) to generate.
        num_images (int): The number of images to generate for the digit.
    """
    if not 0 <= digit <= 9:
        print("Error: Please provide a digit between 0 and 9.")
        return

    # --- Create Output Directory ---
    # Create the folder to save images if it doesn't already exist.
    if not os.path.exists(OUTPUT_FOLDER):
        try:
            os.makedirs(OUTPUT_FOLDER)
            print(f"Created directory: '{OUTPUT_FOLDER}'")
        except OSError as e:
            print(f"Error creating directory '{OUTPUT_FOLDER}': {e}")
            return

    # --- Prompt Definition ---
    # This prompt is carefully crafted to get an MNIST-style image.
    prompt_text = (
        f"A single handwritten digit '{digit}', centered. The style is exactly like the "
        "MNIST dataset: a 28x28 grayscale image with a black background (value 0) and "
        "white foreground (the digit, value 255). The digit should be clear and distinct."
    )
    
    print(f"\nGenerating {num_images} images for the digit '{digit}'...")

    # --- Image Generation Loop ---
    for i in range(num_images):
        print(f"  - Generating image {i + 1}/{num_images}...")
        
        # --- API Payload ---
        # This is the data we send to the Google AI API.
        payload = {
            "instances": [{"prompt": prompt_text}],
            "parameters": {"sampleCount": 1}
        }

        # --- API Call ---
        try:
            response = requests.post(
                f"{API_URL}?key={API_KEY}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60 # Set a timeout for the request
            )
            # Raise an error if the request was unsuccessful (e.g., 4xx or 5xx status codes).
            response.raise_for_status()

            # --- Process Response ---
            result = response.json()
            
            if "predictions" in result and len(result["predictions"]) > 0 and "bytesBase64Encoded" in result["predictions"][0]:
                # The image data is returned as a base64 encoded string. We need to decode it.
                base64_data = result["predictions"][0]["bytesBase64Encoded"]
                image_data = base64.b64decode(base64_data)
                
                # Use the Pillow library to open the image data from bytes.
                image = Image.open(io.BytesIO(image_data))
                
                # --- Save Image ---
                # Create a unique filename for each image.
                file_path = os.path.join(OUTPUT_FOLDER, f"digit_{digit}_image_{i + 1}.png")
                image.save(file_path, "PNG")
                print(f"    -> Successfully saved to '{file_path}'")

            else:
                print("    -> Error: The API response did not contain valid image data.")
                print(f"    -> Response: {result}")

        except requests.exceptions.RequestException as e:
            print(f"    -> An error occurred while calling the API: {e}")
            # Stop if one request fails, as subsequent ones will likely fail too.
            break 
        except (KeyError, IndexError) as e:
            print(f"    -> Error processing the API response. Unexpected structure: {e}")
            break


def main():
    """
    Main function to run the script. Prompts the user for input.
    """
    print("--- Handwritten Digit Image Generator ---")
    
    while True:
        try:
            user_input = input("Enter the digit you want to generate (0-9), or 'q' to quit: ")
            if user_input.lower() == 'q':
                break
            
            selected_digit = int(user_input)
            generate_digit_images(selected_digit, num_images=5)

        except ValueError:
            print("Invalid input. Please enter a number between 0 and 9.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
