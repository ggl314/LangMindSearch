import os

from dotenv import load_dotenv
from lagent.llms import (
    GPTAPI,
    INTERNLM2_META,
    HFTransformerCasualLM,
    LMDeployClient,
    LMDeployServer,
)

internlm_server = dict(
    type=LMDeployServer,
    path="internlm/internlm2_5-7b-chat",
    model_name="internlm2_5-7b-chat",
    meta_template=INTERNLM2_META,
    top_p=0.8,
    top_k=1,
    temperature=0,
    max_new_tokens=8192,
    repetition_penalty=1.02,
    stop_words=["<|im_end|>", "<|action_end|>"],
)

internlm_client = dict(
    type=LMDeployClient,
    model_name="internlm2_5-7b-chat",
    url="http://127.0.0.1:23333",
    meta_template=INTERNLM2_META,
    top_p=0.8,
    top_k=1,
    temperature=0,
    max_new_tokens=8192,
    repetition_penalty=1.02,
    stop_words=["<|im_end|>", "<|action_end|>"],
)

internlm_hf = dict(
    type=HFTransformerCasualLM,
    path="internlm/internlm2_5-7b-chat",
    meta_template=INTERNLM2_META,
    top_p=0.8,
    top_k=None,
    temperature=1e-6,
    max_new_tokens=8192,
    repetition_penalty=1.02,
    stop_words=["<|im_end|>", "<|action_end|>"],
)
# openai_api_base needs to fill in the complete chat api address, such as: https://api.openai.com/v1/chat/completions
gpt4 = dict(
    type=GPTAPI,
    model_type="gpt-4-turbo",
    key=os.environ.get("OPENAI_API_KEY", "YOUR OPENAI API KEY"),
    api_base=os.environ.get("OPENAI_API_BASE",
                            "https://api.openai.com/v1/chat/completions"),
)

url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
qwen = dict(
    type=GPTAPI,
    model_type="qwen-max-longcontext",
    key=os.environ.get("QWEN_API_KEY", "YOUR QWEN API KEY"),
    api_base=url,
    meta_template=[
        dict(role="system", api_role="system"),
        dict(role="user", api_role="user"),
        dict(role="assistant", api_role="assistant"),
        dict(role="environment", api_role="system"),
    ],
    top_p=0.8,
    top_k=1,
    temperature=0,
    max_new_tokens=4096,
    repetition_penalty=1.02,
    stop_words=["<|im_end|>", "<|action_end|>"],
)

internlm_silicon = dict(
    type=GPTAPI,
    model_type="internlm/internlm2_5-7b-chat",
    key=os.environ.get("SILICON_API_KEY", "YOUR SILICON API KEY"),
    api_base="https://api.siliconflow.cn/v1/chat/completions",
    meta_template=[
        dict(role="system", api_role="system"),
        dict(role="user", api_role="user"),
        dict(role="assistant", api_role="assistant"),
        dict(role="environment", api_role="system"),
    ],
    top_p=0.8,
    top_k=1,
    temperature=0,
    max_new_tokens=8192,
    repetition_penalty=1.02,
    stop_words=["<|im_end|>", "<|action_end|>"],
)

# GPTAPI subclass that merges consecutive system messages into one.
#
# MindSearch's InternLMToolAggregator always produces two back-to-back system
# messages: one from the agent's `template` (the date string) and one from
# format_instruction() (the graph / plugin prompt).  Qwen's chatml Jinja
# template raises "System message must be at the beginning" if it encounters
# a system message at any index other than 0, so we must collapse them before
# the request is sent.
class _QwenLlamaCppAPI(GPTAPI):
    @staticmethod
    def _merge_system(messages):
        merged = []
        for msg in messages:
            if (msg.get('role') == 'system' and merged
                    and merged[-1].get('role') == 'system'):
                merged[-1] = dict(merged[-1],
                                  content=merged[-1]['content'] + '\n\n' + msg['content'])
            else:
                merged.append(msg)
        return merged

    def _stream_chat(self, messages, **gen_params):
        return super()._stream_chat(self._merge_system(messages), **gen_params)


# llama.cpp via llama-server (OpenAI-compatible endpoint) with a Qwen model.
# model_type must start with 'gpt' so GPTAPI uses the standard OpenAI request
# format (the 'qwen' branch targets DashScope's proprietary format instead).
# The actual model name passed to llama-server is irrelevant; it ignores it.
llamacpp_server = dict(
    type=_QwenLlamaCppAPI,
    model_type="gpt-local",
    key="sk-no-key-required",
    api_base="http://localhost:8000/v1/chat/completions",
    meta_template=[
        dict(role="system", api_role="system"),
        dict(role="user", api_role="user"),
        dict(role="assistant", api_role="assistant"),
        dict(role="environment", api_role="user"),
    ],
    top_p=0.8,
    temperature=0.0,
    max_new_tokens=8192,
    stop_words=["<|im_end|>", "<|action_end|>"],
)

# Async variant — init_agent() prepends 'Async' to the class name automatically
# when use_async=True, resolving to lagent.llms.Async_QwenLlamaCppAPI.
# If that fails (lagent doesn't know the custom class), use sync mode instead.
llamacpp_server_async = dict(
    type=_QwenLlamaCppAPI,
    model_type="gpt-local",
    key="sk-no-key-required",
    api_base="http://localhost:8000/v1/chat/completions",
    meta_template=[
        dict(role="system", api_role="system"),
        dict(role="user", api_role="user"),
        dict(role="assistant", api_role="assistant"),
        dict(role="environment", api_role="user"),
    ],
    top_p=0.8,
    temperature=0.0,
    max_new_tokens=8192,
    stop_words=["<|im_end|>", "<|action_end|>"],
)
