import os
import torch
from PIL import Image
import numpy as np
import requests
import io
import base64
import gzip
import logging
import pillow_avif # Ensure this is installed if you're using AVIF
import time

from server import PromptServer

class AQ_SendImageToAPI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "api_endpoint": ("STRING", {"default": "https://yourapiendpoint.com/upload"}),
                "api_key": ("STRING", {"default": "YOUR_API_KEY"}),
                "batch_size": ("INT", {"default": 10, "min": 1, "max": 100}),
                "image_format": (["PNG", "JPEG", "WEBP", "AVIF"], {"default": "PNG"}),
                "quality": ("INT", {"default": 85, "min": 1, "max": 100}),
            },
            "optional": {
                "image_name_prefix": ("STRING", {"default": "image"}),
                "jobId": ("STRING", {"default": ""}),
                "userId": ("STRING", {"default": ""}),
                "compression_level": ("INT", {"default": 6, "min": 0, "max": 9}),
                "send_gzipped": ("BOOLEAN", {"default": True}),
                "show_progress_logs": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("instruction", "result",)
    FUNCTION = "send_images_to_api"
    OUTPUT_NODE = True
    CATEGORY = "Aquasite/API"

    def send_images_to_api(self, images, api_endpoint, api_key, batch_size, image_format, quality, 
                          image_name_prefix="image", jobId="", userId="", compression_level=6, 
                          send_gzipped=True, show_progress_logs=True):
        instruction_string = """
        **AQ_SendImageToAPI Node Instructions**

        Purpose:
        This node sends images to a specified API endpoint. It processes images in batches,
        compresses them, and transmits them to the server.

        Inputs:
        - images: The images to be sent.
        - api_endpoint (URL): The server API endpoint where images will be sent.
        - api_key (String): The authorization key for API access.
        - batch_size (Int): Number of images to send in each batch (1-100).
        - image_format (String): Format for the image ('PNG', 'JPEG', 'WEBP', 'AVIF').
        - quality (Int): Quality for JPEG/WEBP/AVIF formats (1-100).
        - image_name_prefix (String): Optional prefix for image filenames (conceptual).
        - jobId (String): Optional Job ID to associate with these images on the server (conceptual).
        - userId (String): Optional User ID to associate with these images on the server (conceptual).
        - compression_level (Int): Gzip compression level for image data (0-9).
        - send_gzipped (Boolean): Data is currently always gzipped; `compression_level` controls the degree.
        - show_progress_logs (Boolean): Whether to print a summary message to the console.

        (Note: The following parameters are described for completeness but are not all currently active inputs in this version of the node: `image_name_prefix`, `jobId`, `userId`, `send_gzipped` as a direct boolean toggle for gzip.)

        - image_name_prefix (String): Optional prefix for image filenames (conceptual).
        - quality (Int): Corresponds to `quality` for WEBP/JPG/AVIF.
        - jobId (String): Optional Job ID to associate with these images on the server (conceptual).
        - userId (String): Optional User ID to associate with these images on the server (conceptual).
        - send_gzipped (Boolean): Data is currently always gzipped; `compression_level` controls the degree.

        Node Behavior:
        - This is an OUTPUT_NODE. It does not pass data directly to other nodes in the ComfyUI workflow via its main return path. Its primary function is to send data externally.
        - The node provides status updates during processing (e.g., progress of image uploads) via the `update_node_status` method, which communicates with the ComfyUI frontend.

        Output:
        - instruction (String): This instructional text.
        - result (String): A string detailing the outcome of the last API call attempt.
        """
        if images is None or len(images) == 0:
            return (instruction_string, "No images to process.",)

        results = [] # This stores the JSON responses from the API for each batch
        overall_result_string = "No API call attempted yet." # Initialize for the final return
        total_images = len(images)
        processed_images_count = 0
        successful_uploads = 0
        failed_uploads = 0
        
        # Get the client ID for progress updates
        prompt_server = PromptServer.instance
        client_id = prompt_server.client_id if prompt_server else None

        for i in range(0, total_images, batch_size):
            batch_images = images[i:i + batch_size]
            batch_payload = []

            for image_tensor in batch_images:
                # Convert tensor to PIL Image using the working approach
                array = 255.0 * image_tensor.cpu().numpy().squeeze()
                pil_image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

                # Convert image to specified format and compress
                img_byte_arr = io.BytesIO()
                save_params = {}
                
                if image_format == "JPEG":
                    pil_image.save(img_byte_arr, format="JPEG", quality=quality)
                elif image_format == "WEBP":
                    pil_image.save(img_byte_arr, format="WEBP", quality=quality)
                elif image_format == "AVIF":
                    pil_image.save(img_byte_arr, format="AVIF", quality=quality)
                else:  # Default to PNG
                    pil_image.save(img_byte_arr, format="PNG")
                
                image_data = img_byte_arr.getvalue()
                
                # Compress with gzip and encode if requested
                if send_gzipped:
                    compressed_data = gzip.compress(image_data, compresslevel=compression_level)
                    encoded_data = base64.b64encode(compressed_data).decode('utf-8')
                else:
                    encoded_data = base64.b64encode(image_data).decode('utf-8')
                
                batch_payload.append({
                    "image_name": f"{image_name_prefix}_{processed_images_count}.{image_format.lower()}",
                    "image_data": encoded_data,
                    "is_gzipped": send_gzipped
                })

            if not batch_payload:
                continue

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            result_string_for_batch = "No API call attempted for this batch." # Initialize for current batch
            try:
                json_payload = {
                    "images": batch_payload,
                    "jobId": jobId if jobId else None,
                    "userId": userId if userId else None
                }
                response = requests.post(api_endpoint, json=json_payload, headers=headers)
                response.raise_for_status()
                
                # Successfully got a response
                response_data = response.json()
                results.append(response_data) # Store structured response
                successful_uploads += len(batch_payload)
                result_string_for_batch = f"Success: Status {response.status_code} - {response_data}"
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Error sending batch to API: {e}")
                error_info = {"error": str(e), "status_code": e.response.status_code if e.response else 'N/A'}
                results.append(error_info) # Store error info
                failed_uploads += len(batch_payload)
                result_string_for_batch = f"Error: {str(e)}" # Update result string with error
            
            overall_result_string = result_string_for_batch # Update overall result with the latest batch status
            
            processed_images_count += len(batch_payload)
            if client_id:
                # The update_node_status is for UI progress, not for the data output string.
                # It uses counts, not the result_string_for_batch.
                self.update_node_status(client_id, total_images, processed_images_count, successful_uploads, failed_uploads)
        
        final_summary_message = f"Overall Processed: {processed_images_count}/{total_images} images. Successful Uploads: {successful_uploads}, Failed Uploads: {failed_uploads}."
        if show_progress_logs:
            print(final_summary_message) 

        # Final status update to UI, if client_id is available
        if client_id:
             self.update_node_status(client_id, total_images, processed_images_count, successful_uploads, failed_uploads)
            
        return (instruction_string, overall_result_string,)

    def update_node_status(self, client_id, total_images, processed_count, successful_uploads, failed_uploads):
        """Sends progress updates to the ComfyUI client."""
        progress = processed_count / total_images
        # This is a simplified progress update. For more detailed UI updates,
        # you might need to interact with ComfyUI's JavaScript side.
        # The `progress` value could be sent via websockets if `client_id` is available.
        # For now, we'll log it, which might appear in the console where ComfyUI is run.
        
        # Note: Direct UI updates from Python are limited.
        # This function is more of a placeholder for where such logic would go.
        # Actual progress bar updates often require custom front-end code.
        
        # Example of sending a progress event (requires PromptServer and client_id)
        if PromptServer.instance and client_id:
            PromptServer.instance.send_sync("progress", {"value": processed_count, "max": total_images, "sid": client_id}, client_id)
            # You could also send custom messages for successful/failed uploads if your UI handles them
            # For example:
            # PromptServer.instance.send_sync("custom_event_name", {"successful": successful_uploads, "failed": failed_uploads, "sid": client_id}, client_id)
        
        logging.info(f"Progress (client {client_id}): {processed_count}/{total_images} images. Successful: {successful_uploads}, Failed: {failed_uploads}.")