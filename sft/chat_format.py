"""One explicit chat format shared by training and qualitative evaluation."""

# The base Qwen checkpoint is not assumed to ship a chat template. These marker
# tokens are part of the Qwen2.5 tokenizer vocabulary.
CHAT_TEMPLATE = """{%- if messages and messages[0]['role'] != 'system' -%}
{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}
{%- endif -%}
{%- for message in messages -%}
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{ '<|im_start|>assistant\n' }}
{%- endif -%}"""

