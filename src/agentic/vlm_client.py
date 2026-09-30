import base64
import cv2
import os
import numpy as np
from typing import List, Dict, Any, Optional
from openai import OpenAI

class VLMClient:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def _encode_frame_to_base64(self, frame: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(buffer).decode('utf-8')

    def verify_event(self, frames: List[np.ndarray], system_prompt: str, user_question: str) -> Dict[str, Any]:
        if not self.client:
            return{
                "verified": False,
                "confidence": 0.0,
                "reasoning": "OpenAI API key is missing: Skipping VLM verification.",
                "confirmed_state": "UNKNOWN"
            }

        content_payload = [{"type": "text", "text": user_question}]
        step = max(1, len(frames) // 5)
        selected_frames = frames[::step][:5]

        for idx, img in enumerate(selected_frames):
            b64_img = self._encode_frame_to_base64(img)
            content_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}",
                    "detail": "low"
                }
            })

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_payload}
                ],
                max_tokens=300,
                temperature=0.1
            )
            raw_text = response.choices[0].message.content
            return self._parse_vlm_response(raw_text)
        except Exception as e:
            return {
                "verified": False,
                "confidence": 0.0,
                "reasoning": f"VLM API Call Failed: {str(e)}",
                "confirmed_state": "UNKNOWN"
            }

    def _parse_vlm_response(self, response_text: str) -> Dict[str, Any]:
        text = response_text.upper()
        verified = "YES" in text or "VERIFIED" in text
        
        confirmed_state = "UNKNOWN"
        if "BED_EXIT" in text or "OUT_OF_BED" in text:
            confirmed_state = "OUT_OF_BED"
        elif "LYING" in text:
            confirmed_state = "LYING_IN_BED"
        elif "SITTING" in text:
            confirmed_state = "SITTING_ON_BED"

        return {
            "verified": verified,
            "raw_response": response_text,
            "confirmed_state": confirmed_state
        }
        