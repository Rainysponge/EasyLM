import pprint
from functools import partial

import numpy as np
import mlxu
import time
import json

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import random
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
import optax
from transformers import GenerationConfig, FlaxLogitsProcessorList
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.serving import LMServer
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules, tree_apply,
    set_random_seed, get_float_dtype_by_name, make_shard_and_gather_fns,
    with_sharding_constraint, FlaxTemperatureLogitsWarper
)
from EasyLM.models.llama.llama_model import LLaMAConfig, FlaxLLaMAForCausalLM
from EasyLM.models.llama.llama_model_splited_transformer import FlaxLLaMAForCausalLMClient, FlaxLLaMAForCausalLMServer, FlaxLLaMAModule, FlaxLLaMAForCausalLMMid

import spu.utils.distributed as ppd
from contextlib import contextmanager
import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2

from flax.linen.linear import Array
from typing import Any, Optional, Tuple, Union


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mesh_dim='1,-1,1',
    dtype='bf16',
    input_length=1024,
    seq_length=2048,
    top_k=50,
    top_p=1.0,
    do_sample=True,
    num_beams=1,
    add_bos_token=True,
    load_llama_config='',
    load_checkpoint='',
    load_server_checkpoint='/mnt/workspace/huzhanyi/llama/JAX_llama/converted_llama_model/7B_split/checkpoint_server',
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    lm_server=LMServer.get_default_config(),
    jax_distributed=JaxDistributedConfig.get_default_config(),
)


import spu.utils.distributed as ppd

ppd.init(ppd.SAMPLE_NODES_DEF, ppd.SAMPLE_DEVICES_DEF)
with open("/mnt/workspace/huzhanyi/sf_SPU/spu_/examples/python/ml/flax_llama7b/3pc.json", 'r') as file:
    conf = json.load(file)

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True

def hack_softmax(
    x: Array,
    axis: Optional[Union[int, Tuple[int, ...]]] = -1,
    where: Optional[Array] = None,
    initial: Optional[Array] = None,
) -> Array:
    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max

    # exp on large negative is clipped to zero
    b = x > -14
    nexp = jnp.exp(x)

    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)

    return b * (nexp / divisor)


@contextmanager
def hack_softmax_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_softmax = jnn.softmax
    jnn.softmax = hack_softmax
    yield
    # recover back
    jnn.softmax = raw_softmax


def hack_gelu(x: Array) -> Array:
    b0 = x < -4.0
    b1 = x < -1.95
    b2 = x > 3.0
    b3 = b1 ^ b2 ^ True  # x in [-1.95, 3.0]
    b4 = b0 ^ b1  # x in [-4, -1.95)

    # seg1 = a[3] * x^3 + a[2] * x^2 + a[1] * x + a[0]
    # seg2 = b[6] * x^6 + b[4] * x^4 + b[2] * x^2 + b[1] * x + b[0]
    a_coeffs = jnp.array(
        [
            -0.5054031199708174,
            -0.42226581151983866,
            -0.11807612951181953,
            -0.011034134030615728,
        ]
    )
    b_coeffs = jnp.array(
        [
            0.008526321541038084,
            0.5,
            0.3603292692789629,
            0.0,
            -0.037688200365904236,
            0.0,
            0.0018067462606141187,
        ]
    )
    x2 = jnp.square(x)
    x3 = jnp.multiply(x, x2)
    x4 = jnp.square(x2)
    x6 = jnp.square(x3)

    seg1 = a_coeffs[3] * x3 + a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
    seg2 = (
        b_coeffs[6] * x6
        + b_coeffs[4] * x4
        + b_coeffs[2] * x2
        + b_coeffs[1] * x
        + b_coeffs[0]
    )

    ret = b2 * x + b4 * seg1 + b3 * seg2

    return ret


@contextmanager
def hack_gelu_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_gelu = jnn.gelu
    jnn.gelu = hack_gelu
    yield
    # recover back
    jnn.gelu = raw_gelu



def init_model_and_optimizer(rng):
    llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
    hf_model = FlaxLLaMAForCausalLM(
        llama_config,
        input_shape=(1, FLAGS.seq_length),
        seed=FLAGS.seed,
        _do_init=False
    )
    
    params = hf_model.init(rng, jnp.ones((1, FLAGS.seq_length), dtype=jnp.int32))
    optimizer = optax.adam(learning_rate=FLAGS.learning_rate).create(params)
    return hf_model, optimizer

# def text_generation(input_ids, params, token_num=8):
#     config = LLaMAConfig()
#     model = FlaxLLaMAForCausalLM(config=config)
#     for _ in range(token_num):
#         outputs = model(input_ids=input_ids, params=params)
#         next_token_logits = outputs[0][0, -1, :]
#         next_token = jnp.argmax(next_token_logits)
#         input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
#     return input_ids
def text_generation(input_ids, params, token_num=8):
    config = LLaMAConfig()
    model = FlaxLLaMAForCausalLM(config=config)
    for _ in range(token_num):
        outputs = model(input_ids=input_ids, params=params)
        next_token_logits = outputs[0][0, -1, :]
        next_token = jnp.argmax(next_token_logits)
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
    return input_ids



def embeding_generation(input_ids, params):
    config = LLaMAConfig()
    model = FlaxLLaMAForCausalLMClient(config=config)
    smasheddata, attention_mask, position_ids = model(input_ids=input_ids, params=params)
    del model
    return smasheddata, attention_mask, position_ids

def mid_generation(input_ids, params, attention_mask, position_ids):

    config = LLaMAConfig()
    _model = FlaxLLaMAForCausalLMMid(config=config)

    _smasheddata = _model(input_ids=input_ids, params=params, attention_mask=attention_mask, position_ids=position_ids)
    
    return _smasheddata, attention_mask, position_ids



def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    set_random_seed(FLAGS.seed)

    model_path = '/mnt/workspace/huzhanyi/llama/JAX_llama/converted_llama_model/'
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # tokenizer = LLaMAConfig.get_tokenizer(
    #     FLAGS.tokenizer, truncation_side='right', padding_side='right'
    # )
    with jax.default_device(jax.devices("cpu")[0]):
        llama_config = LLaMAConfig.load_config(FLAGS.load_llama_config)
        _, params = StreamingCheckpointer.load_trainstate_checkpoint(
            FLAGS.load_checkpoint, disallow_trainstate=True
        )

    # with jax.default_device(jax.devices("cpu")[0]):
    #     model_client = FlaxLLaMAForCausalLMClient(
    #         llama_config,
    #         input_shape=(1, FLAGS.seq_length),
    #         seed=FLAGS.seed,
    #         _do_init=False
    #     )
    # with jax.default_device(jax.devices("cpu")[0]):
    #     model_client = FlaxLLaMAForCausalLMClient.from_pretrained("/mnt/workspace/huzhanyi/llama/JAX_llama/converted_llama_model/hf_new", from_pt=True)

    with jax.default_device(jax.devices()[0]):
        model_server = FlaxLLaMAForCausalLMServer(
            llama_config,
            input_shape=(1, FLAGS.seq_length),
            seed=FLAGS.seed,
            _do_init=False
        )
        # model_server = FlaxLLaMAForCausalLMServer.from_pretrained("/mnt/workspace/huzhanyi/llama/JAX_llama/converted_llama_model/hf_new", from_pt=True)

    print("model load compelete")


    next_token_text = ""
    input_ids = tokenizer.encode('Hello, my dog is cute and', return_tensors='jax')

    # for key, v in params['params']["transformer"]['h'].items():
    #     print(key, type(key))

    client_params_dict = {
        "transformer":{
            "wte":params['params']["transformer"]["wte"],
            "ln_f": params['params']["transformer"]["ln_f"],
        }
    }

    mid_params_dict = {
        "transformer":{

            "h":{
                "0": params['params']["transformer"]["h"]["0"]
            }
        }
    }

    

    for i in range(16):
        start_time = time.time()
        smasheddata, attention_mask, position_ids = embeding_generation(input_ids=input_ids, params=client_params_dict)
        noise = jax.random.normal(jax.random.PRNGKey(0), shape=smasheddata.shape, dtype=smasheddata.dtype)
        mean=0
        std=0.001
        smasheddata = smasheddata + noise * std + mean
        # print("before write", smasheddata, attention_mask, position_ids)
        with open("./embed.txt", "wb+") as f:
            numpy_array = jax.device_get(smasheddata)  # 将ArrayImpl转换为NumPy数组
            print("numpy_array", numpy_array.shape)
            # 将NumPy数组转换为字符串
            data_str = numpy_array.tobytes()
            f.write(data_str)

        _smasheddata, attention_mask, position_ids = mid_generation(input_ids=smasheddata, params=mid_params_dict, attention_mask=attention_mask, position_ids=position_ids)
        # with hack_softmax_context("hack exp of softmax", enabled=True), hack_gelu_context(
        #     "hack gelu", enabled=True
        # ):
        #     _input_ids = ppd.device("P1")(lambda x: x)(smasheddata)
        #     _params = ppd.device("P2")(lambda x: x)(mid_params_dict)

        #     _smasheddata, attention_mask, position_ids = ppd.device("SPU")(mid_generation)(_input_ids, _params, attention_mask, position_ids)
        #     print("before write", _smasheddata.shape, attention_mask, position_ids)
        #     _smasheddata, attention_mask, position_ids = ppd.get(_smasheddata), ppd.get(attention_mask), ppd.get(position_ids)

        # print("before write", _smasheddata.shape, attention_mask, position_ids)

        # smasheddata = jax.device_put(smasheddata)

        with open("./tmp.txt", "wb+") as f:
            numpy_array = jax.device_get(_smasheddata)  # 将ArrayImpl转换为NumPy数组

            # 将NumPy数组转换为字符串
            data_str = numpy_array.tobytes()
            f.write(data_str)

        
        # print("smasheddata", smasheddata)
        outputs = model_server(input_ids=smasheddata, params=params['params'], attention_mask=attention_mask, position_ids=position_ids)
        
        # outputs = model(input_ids=input_ids, params=params['params'])
        next_token_logits = outputs[0][0, -1, :]
        next_token = jnp.argmax(next_token_logits)
        next_token_text += " " + tokenizer.decode(next_token)
        print(next_token_text)
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
        end_time = time.time()

        loop_duration = end_time - start_time

        print(f"Loop {i+1}: {loop_duration} seconds")
    print(outputs)
    print(len(outputs.logits))
    
    

if __name__ == "__main__":
    mlxu.run(main)