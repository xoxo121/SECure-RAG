import logging
import math
from typing import cast

import torch
from dotenv import load_dotenv
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


class FinGuard:
    def __init__(
        self,
        model_path: str = "ibm-granite/granite-guardian-3.0-2b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.safe_token = "No"
        self.unsafe_token = "Yes"
        self.nlogprobs = 20
        logging.info("Loading Granite Guardian Model...")

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",  # Optimizes device placement
            torch_dtype=torch.bfloat16,  # Use bfloat16 for reduced precision computation for speed
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Risk types supported by Granite Guardian
        self.risk_types = [
            "harm",
            "social_bias",
            "jailbreaking",
            "violence",
            "profanity",
            "sexual_content",
            "unethical_behavior",
            "context_relevance",
        ]

    def _parse_output(self, output, input_len):
        """Parse the model output to determine safety"""
        label, prob_of_risk = None, None

        # Process log probabilities for risk detection
        if self.nlogprobs > 0:
            list_index_logprobs_i = [
                torch.topk(
                    token_i, k=self.nlogprobs, largest=True, sorted=True
                )
                for token_i in list(output.scores)[:-1]
            ]
            if list_index_logprobs_i is not None:
                prob = self._get_probabilities(list_index_logprobs_i)
                prob_of_risk = prob[1]

        res = self.tokenizer.decode(
            output.sequences[:, input_len:][0], skip_special_tokens=True
        ).strip()

        # Checking for unsafe tokens like "Yes" (unsafe) or "No" (safe)
        if self.unsafe_token.lower() in res.lower():
            label = self.unsafe_token
        elif self.safe_token.lower() in res.lower():
            label = self.safe_token
        else:
            label = "Failed"

        return label, prob_of_risk.item() if prob_of_risk is not None else None

    def _get_probabilities(self, logprobs):
        """Calculate probabilities from log probabilities"""
        safe_token_prob = 1e-50
        unsafe_token_prob = 1e-50

        for gen_token_i in logprobs:
            for logprob, index in zip(
                gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0]
            ):
                decoded_token: str = cast(
                    str, self.tokenizer.convert_ids_to_tokens(index)
                )
                if decoded_token.strip().lower() == self.safe_token.lower():
                    safe_token_prob += math.exp(logprob)
                if decoded_token.strip().lower() == self.unsafe_token.lower():
                    unsafe_token_prob += math.exp(logprob)

        probabilities = torch.softmax(
            torch.tensor(
                [math.log(safe_token_prob), math.log(unsafe_token_prob)]
            ),
            dim=0,
        )

        return probabilities

    async def check(self, content: str) -> dict:
        """
        Perform safety assessment using Granite Guardian

        :param content: Text to check
        :return: Dictionary with safety assessment result
        """
        # Quick safety checks for very short content
        if len(content) < 5:
            return {"is_safe": True}

        try:
            # Prepare messages for the model
            messages = [{"role": "user", "content": content}]
            guardian_config = {"risk_name": "harm"}  # Default risk type

            # Tokenize input
            input_ids = cast(
                transformers.tokenization_utils_base.BatchEncoding,
                self.tokenizer.apply_chat_template(
                    messages,
                    guardian_config=guardian_config,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ),
            ).to(self.model.device)
            input_len = input_ids.shape[1]

            # Generate output
            self.model.eval()
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=20,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Parse output for all risk types
            for risk_type in self.risk_types:
                label, _ = self._parse_output(output, input_len)

                # If any risk type returns "unsafe", consider it unsafe
                if label == self.unsafe_token:
                    return {"is_safe": False}

            # If no unsafe risk type detected
            return {"is_safe": True}

        except Exception as e:
            logging.error(f"Safety check error: {e}")
            return {"is_safe": False}
