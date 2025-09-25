import re
import os
from typing import List, Dict, Any
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMInterface:
    def __init__(self, model_path: str, max_tokens: int = 8192, n_threads: int = 8, gpu_layers: int = -1):
        """Initialize the LLM interface with API support (OpenAI or Groq).

        Args:
            model_path (str): Either:
                - "openai:gpt-3.5-turbo" or "openai:gpt-4" for OpenAI
                - "groq:mixtral-8x7b-32768" or "groq:llama3-8b-8192" for Groq
                - Local model path for vLLM (fallback to original)
            max_tokens (int): Maximum context length
        """
        self.max_tokens = max_tokens

        if model_path.startswith("openai:"):
            self.provider = "openai"
            self.model_name = model_path.split(":", 1)[1]
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.api_url = "https://api.openai.com/v1/chat/completions"
            print(f"Using OpenAI API with model: {self.model_name}")

        elif model_path.startswith("groq:"):
            self.provider = "groq"
            self.model_name = model_path.split(":", 1)[1]
            self.api_key = os.environ.get("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            self.api_url = "https://api.groq.com/openai/v1/chat/completions"
            print(f"Using Groq API with model: {self.model_name}")

        else:
            # Fallback to vLLM for local models
            self.provider = "vllm"
            print(f"Using local vLLM with model: {model_path}")
            # Import vLLM only if needed
            import torch
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.6,
                max_model_len=max_tokens,
                swap_space=0,
                trust_remote_code=True,
                dtype=torch.float16,
            )
            self.SamplingParams = SamplingParams

        # Store configuration
        self.config = {
            "model_path": model_path,
            "max_tokens": max_tokens,
        }

    def trim_to_last_sentence(self, text: str) -> str:
        """Return text truncated at the final full sentence boundary."""
        m = re.match(r"^(.*?[.!?][\"')\]]?)\s*$", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ".!?":
                return text[: i + 1].strip()
        return text.strip()

    def generate_response(self, system_prompt: str, user_message: str, conversation_history: str = "") -> str:
        """Generate a response from the LLM using the appropriate API or local model."""

        if self.provider in ["openai", "groq"]:
            # Use API
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # Add conversation history if provided
            if conversation_history:
                # Parse the history (simplified - you may need to adjust based on format)
                for line in conversation_history.split('\n'):
                    if line.startswith("User:"):
                        messages.append({"role": "user", "content": line[5:].strip()})
                    elif line.startswith("AI:"):
                        messages.append({"role": "assistant", "content": line[3:].strip()})

            # Add current user message
            messages.append({"role": "user", "content": user_message})

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 100,  # Keep responses short for voice
                "temperature": 0.7,
                "stream": False
            }

            try:
                response = requests.post(self.api_url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                text = result['choices'][0]['message']['content']
                return self.trim_to_last_sentence(text)
            except Exception as e:
                print(f"API error: {e}")
                return "I'm having trouble connecting to the AI service."

        else:
            # Use vLLM (original implementation)
            prompt = f"""<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>
            {conversation_history}
            <|start_header_id|>user<|end_header_id|>\n{user_message}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>\n"""

            sampling_params = self.SamplingParams(
                temperature=1.0,
                top_p=0.95,
                max_tokens=100,
                repetition_penalty=1.2,
                top_k=200,
                stop=["</s>", "<|endoftext|>", "<<USR>>", "<</USR>>", "<</SYS>>",
                      "<</USER>>", "<</ASSISTANT>>", "<|end_header_id|>", "<<ASSISTANT>>",
                      "<|eot_id|>", "<|im_end|>", "user:", "User:", "user :", "User :"]
            )

            outputs = self.llm.generate(prompt, sampling_params)

            if outputs and len(outputs) > 0:
                text = outputs[0].outputs[0].text
                return self.trim_to_last_sentence(text)
            return ""

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text - simplified for API usage."""
        # Rough approximation: 1 token ≈ 4 characters for English
        return list(range(len(text) // 4))

    def get_token_count(self, text: str) -> int:
        """Return approximate token count."""
        return len(text) // 4

    def batch_generate(self, prompts: List[Dict[str, str]],
                       max_tokens: int = 512,
                       temperature: float = 0.7) -> List[str]:
        """Generate responses for multiple prompts in a batch."""
        results = []
        for p in prompts:
            system = p.get("system", "")
            user = p.get("user", "")
            history = p.get("history", "")
            response = self.generate_response(system, user, history)
            results.append(response)
        return results