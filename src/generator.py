import asyncio
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from typing import Any
import numpy as np
from statistics import mean
from time import perf_counter
from loguru import logger
from tiktoken import Encoding, encoding_for_model, get_encoding
from .templates import ApiReturn, ChatHistory, client



rag_drafting_prompt: str = """Response to the instruction based on the context provided by the user.
Also provide rationale for your response.
Your response should start with "So the answer is".
Ensure that you always provide an answer and do not respond with phrases like "I don't know". If there is not enough information, make an educated guess.
## Instruction: {instruction}

## Evidence: {evidence}
## Chat History: {history} """

class RagDraftingResponse(BaseModel):
    rationale: str = Field(description="Response rationale.")
    response: str = Field(description="Response to the instruction.")


async def rag_drafting_generator(
    model_name: str,
    instruction: str,
    evidence: str,
    chat_history: ChatHistory,
    **kwargs,
) -> tuple[RagDraftingResponse, ApiReturn]:
    
    messages=[  # system
        {'role': 'system', 'content': 'Follow the given examples and answer the question.'},
        {'role': 'system', 'content': 'You are a helpful assistant.'}
    ] + [  # current example
        {
            "role": "system",
            "content": rag_drafting_prompt.format(
                instruction=instruction, evidence=evidence, history=chat_history.get_text()
            ),
        }
    ]
    completion: Any = await client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        response_format=RagDraftingResponse,
        temperature=0.0,
        logprobs=True,
        max_tokens=512,
        **kwargs,
    )
    r = completion.choices[0]
    generation = ApiReturn(
            prompt=instruction,
            text=r.message.parsed.response,
            finish_reason=r.finish_reason,
            model=model_name,
            tokens=r.logprobs.content,
            probs=[np.exp(token.logprob) for token in r.logprobs.content],
            offsets=get_offsets(r.logprobs.content),
            skip_len=0)
    
    # print("draft response : ", generation.text)
    # print("draft rationale : ", r.message.parsed.rationale)
    # print("draft logprobs : ", np.exp(mean(token.logprob for token in r.logprobs.content)))
    return (completion.choices[0].message.parsed,
            np.exp(mean(token.logprob for token in r.logprobs.content)),
            generation)
    
    
rag_verifier_prompt: str = """## Instruction: {instruction}

## Response: {response} 

## Rationale: {rationale}

## Evidence(Ground Truth): {evidence} 

Based on the provided evidence, does the response directly and specifically answer the instruction, providing clear information?

Additionally, is the rationale complete and accurate, without any missing or incorrect information, and consistent with the evidence?

Please respond with "Yes" if both conditions are met.

If the response does not directly answer the instruction, is vague, lacks specificity (e.g., "I cannot determine," "I can't infer," "I don't know"), or is inconsistent with the evidence, please respond with "No" without further evaluation."""

async def rag_verifier_generator(
    model_name: str,
    instruction: str,
    evidence: str,
    response: str,
    rationale: str,
    **kwargs,
) -> tuple[Any, float]:
    encoder: Encoding = encoding_for_model(model_name=model_name)
    completion: Any = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": rag_verifier_prompt.format(
                    instruction=instruction,
                    evidence=evidence,
                    response=response,
                    rationale=rationale,
                ),
            }
        ],
        temperature=0.0,
        logprobs=True,
        max_tokens=2,
        **kwargs,
    )
    veri_response: str = completion.choices[0].message.content
    cond: bool = encoder.encode(text=veri_response.lower()) == encoder.encode(text="yes")
    p_yes: float = (
        np.exp(mean(token.logprob for token in completion.choices[0].logprobs.content))
        if cond
        else 1 - np.exp(mean(token.logprob for token in completion.choices[0].logprobs.content))
    )  # Naive
    
    # print("reponse : ", response)
    # print("p_yes : ", p_yes)
    return (veri_response, p_yes)

direct_generator_prompt: str = """## Instruction: {instruction}

## Chat History: {chat_history} 

Answer to the Instruction based on the context provided by the user. 

If the response in the chat history is complete, respond with "<finish>", and do not provide any other response. 

If the answer is not complete, continue answering the question."""


def get_offsets(tokens) -> list[int]:
    l = 0
    offsets = []
    for i, t in enumerate(tokens):
        offsets.append(l)
        l += len(t.token)
    return offsets

max_generation_len = 128

async def direct_generator(
    model_name: str,
    instruction: str,
    chat_history: ChatHistory,
    **kwargs,
) -> ApiReturn:
    # print(f"direct generator chat_history  : {chat_history.get_text()}")
    messages=[  # system
        {'role': 'system', 'content': 'Follow the given examples and answer the question shortly and clearly.'},
        {'role': 'system', 'content': 'You are a helpful assistant.'}
    ] + [  # current example
        {
            "role": "system",
            "content": direct_generator_prompt.format(
                instruction=instruction,
                chat_history=chat_history.get_text(),
            ),
        }
    ]
    completion: Any = await client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        temperature=0.0,
        logprobs=True,
        max_tokens=max_generation_len,
        **kwargs,
    )
    r = completion.choices[0]
    generation = ApiReturn(
            prompt=instruction,
            text=r.message.content,
            finish_reason=r.finish_reason,
            model=model_name,
            tokens=r.logprobs.content,
            probs=[np.exp(token.logprob) for token in r.logprobs.content],
            offsets=get_offsets(r.logprobs.content),
            skip_len=0)
        
    return generation

def check_finish(response: str):
    if response[-8:] == "<finish>":
        print("Answer is finished")
        return True
    return False