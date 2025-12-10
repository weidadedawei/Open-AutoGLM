import base64
import io
import gc
from typing import Any

from PIL import Image

import mlx.core as mx
import mlx_vlm
from phone_agent.model.client import ModelConfig, ModelResponse


class MLXModelClient:
    """
    Client for running local inference using MLX (Apple Silicon).
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        print(f"⏳ Loading local model '{config.model_name}' (this may take a moment)...")
        # Load model and processor
        # trust_remote_code=True is often needed for custom models like GLM-4V
        self.model, self.processor = mlx_vlm.load(
            config.model_name,
            trust_remote_code=True,
            # Optimization flags passed to AutoProcessor/AutoTokenizer via kwargs
            fix_mistral_regex=True,
            use_fast=True,
        )
        print("✅ Model loaded successfully.")

    def request(self, messages: list[dict[str, Any]]) -> ModelResponse:
        """
        Send a request to the local model.
        """
        # 1. Extract images and format prompt
        formatted_messages = []
        images = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            new_content = []
            if isinstance(content, str):
                new_content = content
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        new_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        # Handle base64 image
                        image_url = item["image_url"]["url"]
                        if image_url.startswith("data:image"):
                            # Check format: data:image/png;base64,...
                            header, encoded = image_url.split(",", 1)
                            image_data = base64.b64decode(encoded)
                            image = Image.open(io.BytesIO(image_data))
                            
                            # Optimization: Resize large images to reduce token count
                            # Max dimension 1024 is usually sufficient for UI tasks
                            image = self._resize_image(image, max_size=1024)
                            
                            images.append(image)
                            new_content.append({"type": "image"})
            
            formatted_messages.append({"role": role, "content": new_content})

        # 2. Apply Chat Template
        prompt = self.processor.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 3. Generate
        output = mlx_vlm.generate(
            self.model,
            self.processor,
            prompt=prompt,
            image=images if images else None,
            max_tokens=self.config.max_tokens,
            temp=self.config.temperature,
            kv_bits=8, # Enable 8-bit KV cache quantization for GPU bandwidth optimization
            verbose=False 
        )

        # 4. Parse Response
        raw_content = output.text.strip()
        thinking, action = self._parse_response(raw_content)
        
        # Optimization: Clear cache and gc to free memory
        mx.clear_cache()
        gc.collect()

        return ModelResponse(thinking=thinking, action=action, raw_content=raw_content)

    def _resize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """Resize image while maintaining aspect ratio if it exceeds max_size."""
        width, height = image.size
        if width <= max_size and height <= max_size:
            return image
            
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
            
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _parse_response(self, content: str) -> tuple[str, str]:
        """
        Parse the model response into thinking and action parts.
        """
        if "<answer>" not in content:
            return "", content

        parts = content.split("<answer>", 1)
        thinking = parts[0].replace("<think>", "").replace("</think>", "").strip()
        action = parts[1].replace("</answer>", "").strip()

        return thinking, action
