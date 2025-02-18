"""
openvla.py

PyTorch Module defining OpenVLA as a lightweight wrapper around a PrismaticVLM; defines custom logic around
discretizing actions with the ActionTokenizer.
"""

from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import time
import torch
from PIL.Image import Image as Img
from transformers import LlamaTokenizerFast
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.datasets import AUX_QUESTIONS_PROMPT

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    def __init__(
        self,
        *args,
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        action_tokenizer: ActionTokenizer,
        aux_context_freq: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.norm_stats = norm_stats
        self.action_tokenizer = action_tokenizer
        self.aux_context_freq = aux_context_freq
        self._aux_cache = {}
        self._aux_cache_counter = 0
        self._last_instruction = None

    @torch.inference_mode()
    def predict_action(
        self,
        image: Union[Img, List[Img]],
        instruction: str,
        unnorm_key: Optional[str] = None,
        aux_task_types: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Dict[str, Union[str, np.ndarray]]:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action
        (de-tokenizes), along with auxiliary information if provided.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics
        @param aux_questions: Optional list of auxiliary questions to ask before the final action question
        @return Dictionary containing auxiliary answers and the unnormalized (continuous) action vector
        """
        image_transform, tokenizer = self.vision_backbone.get_image_transform(), self.llm_backbone.tokenizer

        # Preprocess Image
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Build VLA Prompt with Auxiliary Q&A
        prompt_builder = self.get_prompt_builder()

        # Calculate total max tokens: for each aux Q&A pair, set a reasonable max (e.g., 50 tokens), plus action tokens

        output = {}

        if aux_task_types:
            # Check if we need to refresh the cache
            should_refresh = (
                self._aux_cache_counter % self.aux_context_freq == 0
                or instruction != self._last_instruction
                or not self._aux_cache
            )

            if should_refresh:
                # Original aux Q&A logic
                aux_answers = {}
                for aux_type in aux_task_types:
                    aux_prompt = AUX_QUESTIONS_PROMPT[aux_type] + f" {instruction.lower()}?"
                    prompt_builder.add_turn(role="human", message=aux_prompt)
                    
                    # get response from model
                    prompt_text = prompt_builder.get_prompt()
                    input_ids = self.process_prompt_text(tokenizer, prompt_text)

                    # Use autocast to ensure consistent dtype
                    autocast_dtype = self.llm_backbone.half_precision_dtype
                    with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
                        generated_ids = super(PrismaticVLM, self).generate(
                            input_ids=input_ids,                            # Shape: [1, seq]
                            pixel_values=pixel_values,                      # Shape: [1, (opt T,) 3, res, res] or Dict[str, ...]
                            max_new_tokens=250,
                            eos_token_id=151645,
                            **kwargs
                        )
                    generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :])
                    aux_answers[aux_type] = generated_text
                    prompt_builder.add_turn(role="gpt", message=generated_text)
                    
                # Update cache
                self._aux_cache = aux_answers
                self._last_instruction = instruction
            else:
                # Use cached answers
                aux_answers = self._aux_cache
                for aux_type, answer in aux_answers.items():
                    prompt_builder.add_turn(role="human", message=AUX_QUESTIONS_PROMPT[aux_type] + f" {instruction.lower()}?")
                    prompt_builder.add_turn(role="gpt", message=answer)

            self._aux_cache_counter += 1
            output.update(aux_answers)
        

        # Add Final Action Prompt
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")

        prompt_text = prompt_builder.get_prompt()

        input_ids = self.process_prompt_text(tokenizer, prompt_text)
        total_max_tokens = self.get_action_dim(unnorm_key)

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, (opt T,) 3, res, res] or Dict[str, ...]
                max_new_tokens=total_max_tokens,
                eos_token_id=151645,
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        output['action'] = actions
        if aux_task_types:
            if "bbox" in aux_task_types:
                try:
                    output['bbox'] = self.parse_bbox_string(output['bbox'])
                except Exception as e:
                    overwatch.error(f"Error parsing bbox string: {e}")
                    output['bbox'] = {}
            if "ee_pose_2D" in aux_task_types:
                try:
                    output['ee_pose_2D'] = self.parse_ee_pose_2d_string(output['ee_pose_2D'])
                except Exception as e:
                    overwatch.error(f"Error parsing ee_pose_2D string: {e}")
                    output['ee_pose_2D'] = []
            
        return output

    def process_prompt_text(self, tokenizer, prompt_text: str):
        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            if not torch.all(input_ids[:, -1] == 29871):  # Updated Token ID
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
        elif isinstance(tokenizer, Qwen2TokenizerFast):
            pass
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")
        
        return input_ids

    def split_responses(self, generated_text: str, aux_questions: Optional[List[str]]) -> Dict[str, str]:
        """
        Split the generated text into auxiliary answers and final action.

        @param generated_text: The complete generated text from the model
        @param aux_questions: List of auxiliary questions

        @return Dictionary with auxiliary answers and final action
        """
        if not aux_questions:
            return {'final_action': generated_text}

        aux_answers = {}
        remaining_text = generated_text
        for idx, aux_q in enumerate(aux_questions, 1):
            # Define delimiters based on conversation structure
            answer_prefix = f"Assistant {idx}:"
            delimiter = answer_prefix

            # Split based on the answer prefix
            if delimiter in remaining_text:
                answer_split = remaining_text.split(delimiter, 1)
                if len(answer_split) == 2:
                    answer, remaining_text = answer_split
                    aux_answers[f'answer_{idx}'] = answer.strip()
                else:
                    raise ValueError(f"Expected answer delimiter '{delimiter}' not found.")
            else:
                raise ValueError(f"Expected answer prefix '{delimiter}' not found in generated text.")

        final_action = remaining_text.strip()
        aux_answers['final_action'] = final_action

        return {'aux_answers': aux_answers}

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]

    def parse_bbox_string(self, bbox_str: str) -> Dict[str, List[float]]:
        """
        Parse bbox string from model output into a dictionary mapping object names to their bounding boxes.
        
        @param bbox_str: String containing bbox predictions in format "name: [x1, y1, x2, y2], ..."
        @return Dictionary mapping object names to bbox coordinates [x1, y1, x2, y2]
        """
        # Remove the <|im_end|> token if present
        bbox_str = bbox_str.replace('<|im_end|>', '').strip()
        
        # Split into individual bbox predictions
        bbox_entries = bbox_str.split('], ')
        
        bbox_dict = {}
        for entry in bbox_entries:
            # Split name and coordinates
            name, coords = entry.split(': [')
            
            # Clean up coordinates and convert to float list
            coords = coords.rstrip(']').split(', ')
            bbox_dict[name] = [float(coord) for coord in coords]
            
        return bbox_dict

    def parse_ee_pose_2d_string(self, pose_str: str) -> List[Tuple[float, float]]:
        """
        Parse ee_pose_2D string from model output into a list of 2D coordinates.
        
        @param pose_str: String containing pose predictions in format "[(x1, y1), (x2, y2), ...]"
        @return List of tuples containing (x, y) coordinates
        """
        # Remove the <|im_end|> token if present
        pose_str = pose_str.replace('<|im_end|>', '').strip()
        
        # Remove outer brackets and split into coordinate pairs
        pose_str = pose_str.strip('[]')
        coord_pairs = pose_str.split('), (')
        
        poses = []
        for pair in coord_pairs:
            # Clean up coordinates and convert to float tuple
            pair = pair.strip('()').split(', ')
            poses.append((float(pair[0]), float(pair[1])))
            
        return poses
