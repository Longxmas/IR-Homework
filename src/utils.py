from typing import Any, List, Dict
import random
import time
import os
import logging
import copy
import string
import asyncio
import openai
from openai import AsyncOpenAI
logging.basicConfig(level=logging.INFO)

client = AsyncOpenAI()

class Utils:
    punctuations = set(string.punctuation)

    @classmethod
    def is_chat(cls, model: str):
        return True

    @classmethod
    def is_code(cls, model: str):
        return 'code' in model

    @classmethod
    def no_stop(cls, model: str):
        return True


class NoKeyAvailable(Exception):
    pass


def retry_with_exponential_backoff(
    func,
    max_reqs_per_min: int = 0,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    errors_to_catch: tuple = (NoKeyAvailable),
    errors_to_raise: tuple = (),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        # initialize variables
        is_code_model = Utils.is_chat(kwargs['model'])
        mrpm = max_reqs_per_min
        mrpm = mrpm or (15 if is_code_model else 1000)
        const_delay = 60 / mrpm
        delay = initial_delay
        num_retries = 0

        # loop until a successful response or max_retries is hit or an exception is raised
        while True:
            # initialize key-related variables
            api_key = get_key_func = return_key_func = None
            forbid_key = False

            try:
                # get key
                _kwargs = copy.deepcopy(kwargs)
                if 'api_key' in kwargs:
                    ori_api_key = kwargs['api_key']
                    if type(ori_api_key) is tuple:  # get a key through a call
                        get_key_func, return_key_func = ori_api_key
                        api_key = get_key_func()
                    else:  # a specified key
                        api_key = ori_api_key or os.getenv('OPENAI_API_KEY')
                    _kwargs['api_key'] = api_key

                # query API
                start_t = time.time()
                logging.info(f'API call start: {_kwargs.get("api_key", "")[-5:]}')
                results = func(*args, **_kwargs)
                logging.info(f'API call end: {_kwargs.get("api_key", "")[-5:]}')
                return results

            # retry on specific errors
            except errors_to_catch as e:
                # check if the key is useless
                if hasattr(e, 'json_body') and e.json_body is not None and 'error' in e.json_body and 'type' in e.json_body['error'] and e.json_body['error']['type'] == 'insufficient_quota':  # quota error
                    logging.info(f'NO QUOTA: {api_key[-5:]}')
                    forbid_key = True
                if hasattr(e, 'json_body') and e.json_body is not None and 'error' in e.json_body and 'type' in e.json_body['error'] and e.json_body['error']['code'] == 'account_deactivated':  # ban error
                    logging.info(f'BAN: {api_key[-5:]}')
                    forbid_key = True

                # check num of retries
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f'maximum number of retries ({max_retries}) exceeded.')

                # incremental delay
                delay *= exponential_base * (1 + jitter * random.random())
                logging.info(f'retry on {e}, sleep for {const_delay + delay}')
                time.sleep(const_delay + delay)

            # raise on specific errors
            except errors_to_raise as e:
                raise e

            # raise exceptions for any errors not specified
            except Exception as e:
                raise e

            finally:  # return key if necessary
                if api_key is not None and return_key_func is not None:
                    end_t = time.time()
                    return_key_func(api_key, time_spent=end_t - start_t, forbid=forbid_key)

    return wrapper


async def async_chatgpt(
    *args,
    messages: List[List[Dict[str, Any]]],
    **kwargs,
) -> List[str]:
    # async_responses = [
    #     openai.ChatCompletion.acreate(
    #         *args,
    #         messages=x,
    #         **kwargs,
    #     )
    #     for x in messages
    # ]
    async_responses = [
        client.beta.chat.completions.parse(
            *args,
            messages=x,
            logprobs=True,
            **kwargs,
        )  for x in messages
    ]    
    return await asyncio.gather(*async_responses)

async def async_chatgpt_single(
    *args,
    message: List[Dict[str, Any]],
    **kwargs,
) -> str:
    async_response = client.beta.chat.completions.parse(
        *args,
        logprobs=True,
        **kwargs,
    )
    return await async_response

@retry_with_exponential_backoff
def openai_api_call(*args, **kwargs):
    model = kwargs['model']
    is_chat_model = Utils.is_chat(model)
    if is_chat_model:
        if len(kwargs['messages']) <= 0:
            return []
        if type(kwargs['messages'][0]) is list:  # batch request
            return asyncio.run(async_chatgpt(*args, **kwargs))
        else:
            return asyncio.run(async_chatgpt_single(*args, message = kwargs['messages'],**kwargs))
    else:
        return openai.Completion.create(*args, **kwargs)