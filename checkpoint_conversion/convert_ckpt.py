import argparse
import json
import os

import torch
import torch_xla.utils.serialization as xser

from training_utils import create_partition


def merge_llama_tp_checkpoints(args):
    full_model = {}
    if args.kv_size_multiplier > 1:
        with open(args.config, "r") as f:
            config = json.load(f)
        q_heads = config["num_attention_heads"]
        kv_heads = config["num_key_value_heads"]
        head_dim = config["hidden_size"] // q_heads

    for tp_rank in range(args.tp_size):
        for pp_rank in range(args.pp_size):
            if args.load_xser:
                partial_state = load_partial_xser(args, tp_rank, pp_rank)
            else:
                partial_state = load_partial_no_xser(args, tp_rank, pp_rank)
            if args.model_key is not None and args.model_key in partial_state:
                partial_state = partial_state[args.model_key]
            for name, param in partial_state.items():
                if "qkv_proj" in name:
                    # qkv_proj would be a key if we are using the QKVLinear layer
                    partition_dim = 0
                    if name not in full_model:
                        full_model[name] = []
                    full_model[name].append(param)
                    if tp_rank == (args.tp_size - 1):
                        full_weight = torch.cat(full_model[name], dim=partition_dim)
                        if "weight_k" in name or "weight_v" in name:
                            # If kv_multiplier is set, the kv heads are repeated. So we need to
                            # take only the first chunk
                            full_model[name] = torch.chunk(full_weight, args.kv_size_multiplier)[0]
                        else:
                            # Since we do the replication of KV heads, the Q heads are placed as:
                            # Q0Q1Q8Q9...Q2Q3Q10Q11...
                            # Hence when creating the merged checkpoint, we need to bring the Q heads in order.
                            q_weights = full_weight.view(q_heads, head_dim, -1)
                            q_weights_shape = q_weights.size()
                            q_weights = q_weights.view(
                                -1, q_heads // (kv_heads * args.kv_size_multiplier), head_dim, q_weights_shape[-1]
                            )
                            weight_splits = []
                            indicies = torch.arange(0, args.tp_size // kv_heads) * kv_heads
                            for i in range(kv_heads):
                                weight_splits.append(q_weights[indicies + i].view(-1, q_weights_shape[-1]))
                            full_weight = torch.cat(weight_splits, dim=0)
                            full_model[name] = full_weight
                elif (
                    "embed_tokens" in name
                    or "q_proj" in name
                    or "k_proj" in name
                    or "v_proj" in name
                    or "o_proj" in name
                    or "down_proj" in name
                    or "lm_head" in name
                ):
                    partition_dim = 1 if ("o_proj" in name or "down_proj" in name) else 0
                    if name not in full_model:
                        full_model[name] = []
                    full_model[name].append(param)
                    if tp_rank == (args.tp_size - 1):
                        full_weight = torch.cat(full_model[name], dim=partition_dim)
                        full_model[name] = full_weight
                elif "gate_up_proj" in name:
                    partition_dim = 0
                    dim_size = param.size()[partition_dim] // 2
                    gate_proj_name = name.replace("gate_up_proj", "gate_proj")
                    up_proj_name = name.replace("gate_up_proj", "up_proj")
                    gate_proj_weight = param.narrow(partition_dim, 0, dim_size).detach().clone()
                    up_proj_weight = param.narrow(partition_dim, dim_size, dim_size).detach().clone()
                    if gate_proj_name not in full_model:
                        full_model[gate_proj_name] = []
                    if up_proj_name not in full_model:
                        full_model[up_proj_name] = []
                    full_model[gate_proj_name].append(gate_proj_weight)
                    full_model[up_proj_name].append(up_proj_weight)
                    if tp_rank == (args.tp_size - 1):
                        full_gate_proj_weight = torch.cat(full_model[gate_proj_name], dim=partition_dim)
                        full_up_proj_weight = torch.cat(full_model[up_proj_name], dim=partition_dim)
                        full_model[gate_proj_name] = full_gate_proj_weight
                        full_model[up_proj_name] = full_up_proj_weight
                else:
                    if name not in full_model:
                        full_model[name] = param
    return full_model


def translate_llama_full_state_dict_to_tp(
    full_state, tp_size, tp_rank, pp_size, pp_rank, partitions, kv_size_multiplier, config_json
):
    partial_state = {}
    for name, full_p in full_state.items():
        ##################### PP Slice #########################################
        # Embedding only in first PP
        if pp_rank != 0 and "embed_tokens" in name:
            continue
        # LMhead and final layer norm only in last PP rank
        if pp_rank != pp_size - 1 and ("lm_head" in name or "model.norm.weight" in name):
            continue
        if "layers" in name:
            layer_idx = int(name.split(".")[2])
            pre_layer_cut = int(partitions[pp_rank - 1].split(".")[2]) if pp_rank > 0 else -10000000
            current_layer_cut = int(partitions[pp_rank].split(".")[2]) if pp_rank < pp_size - 1 else 10000000
            if layer_idx <= pre_layer_cut or layer_idx > current_layer_cut:
                continue

        ##################### TP Slice #########################################
        if ("q_proj" in name or "k_proj" in name or "v_proj" in name or "qkv_proj" in name) and kv_size_multiplier > 1:
            with open(config_json, "r") as f:
                config = json.load(f)
            q_heads = config["num_attention_heads"]
            kv_heads = config["num_key_value_heads"]
            head_dim = config["hidden_size"] // q_heads
            if "k_proj" in name or ".v_proj" in name or "weight_k" in name or "weight_v" in name:
                repeated_kv = full_p.repeat(kv_size_multiplier, 1)

                dim_size = repeated_kv.size()[0]
                assert dim_size % tp_size == 0, "0th dim after KV replication is not divisible by tp_size"
                partition_size = dim_size // tp_size
                with torch.no_grad():
                    to_load = repeated_kv.narrow(0, tp_rank * partition_size, partition_size).detach().clone()
                    # When kv_multiplier is greater than 1, it means we are using the qkv_linear layer.
                    # Hence the key name needs to be adjusted.
                    name = (
                        ".".join(name.split(".")[:-2])
                        + ".qkv_proj."
                        + ("weight_k" if "weight_k" in name or "k_proj" in name else "weight_v")
                    )
                    partial_state[name] = to_load
            else:
                # When GQAQKV linear with kv_multiplier is used, we need to reshuffle the order of Q heads so that
                # they interact with the right KV heads.
                q_weights = full_p.view(q_heads, head_dim, -1)
                q_weights_shape = q_weights.size()
                q_weights = q_weights.view(
                    -1, q_heads // (kv_heads * kv_size_multiplier), head_dim, q_weights_shape[-1]
                )
                weight_splits = []
                indicies = torch.arange(0, kv_heads) * tp_size // kv_heads
                for i in range(tp_size // kv_heads):
                    weight_splits.append(q_weights[indicies + i])
                q_weights = torch.cat(weight_splits, dim=0)
                with torch.no_grad():
                    to_load = q_weights[tp_rank].view(-1, q_weights_shape[-1])
                    name = ".".join(name.split(".")[:-2]) + ".qkv_proj.weight_q"
                    print(name)
                    partial_state[name] = to_load

        elif (
            "embed_tokens" in name
            or "q_proj" in name
            or "k_proj" in name
            or "v_proj" in name
            or "o_proj" in name
            or "down_proj" in name
            or "lm_head" in name
        ):
            # parallel embedding or ColumnParallelLinear, parallel on 0th dim
            # RowParallelLinear parallel on 1st dim
            partition_dim = 1 if ("o_proj" in name or "down_proj" in name) else 0
            dim_size = full_p.size()[partition_dim]
            assert dim_size % tp_size == 0, "vocab size is not divisiable"
            partition_size = dim_size // tp_size
            with torch.no_grad():
                to_load = full_p.narrow(partition_dim, tp_rank * partition_size, partition_size).detach().clone()
                partial_state[name] = to_load
        elif "gate_proj" in name or "up_proj" in name:
            # ColumnParallelLinear
            partition_dim = 0
            dim_size = full_p.size()[partition_dim]
            assert dim_size % tp_size == 0, "vocab size is not divisiable"
            partition_size = dim_size // tp_size
            with torch.no_grad():
                to_load = full_p.narrow(partition_dim, tp_rank * partition_size, partition_size).detach().clone()
            token = "gate_proj" if "gate_proj" in name else "up_proj"
            updated_name = name.replace(token, "gate_up_proj")
            if updated_name in partial_state:
                if token == "gate_proj":
                    partial_state[updated_name] = torch.cat([to_load, partial_state[updated_name]], dim=0)
                else:
                    partial_state[updated_name] = torch.cat([partial_state[updated_name], to_load], dim=0)
            else:
                partial_state[updated_name] = to_load
        else:
            # no TP
            partial_state[name] = full_p
    return partial_state


# Save Load Entries
def load_full(args):
    full_state = torch.load(args.input_dir)
    return full_state


# def load_partial_xser(args, tp_rank, pp_rank):
#     load_dir = os.path.join(args.input_dir, "tp_rank_{:02d}_pp_rank_{:02d}".format(tp_rank, pp_rank))
#     partial_state = xser.load(load_dir)
#     return partial_state

# def load_partial_xser(args, tp_rank, pp_rank):
#     # load_dir = os.path.join(args.input_dir, "tp_rank_{:02d}_pp_rank_{:02d}".format(tp_rank, pp_rank))
#     load_dir = os.path.join(args.input_dir, "dp_rank_00_tp_rank_{:02d}_pp_rank_{:02d}.pt".format(tp_rank, pp_rank))
#     partial_state = xser.load(load_dir)
#     return partial_state

def load_partial_xser(args, tp_rank, pp_rank):
    # load_dir = os.path.join(args.input_dir, "tp_rank_{:02d}_pp_rank_{:02d}".format(tp_rank, pp_rank))
    load_dir = os.path.join(args.input_dir, "dp_rank_00_tp_rank_{:02d}_pp_rank_{:02d}.pt".format(tp_rank, pp_rank))
    partial_state = xser.load(load_dir)
    return partial_state

def load_partial_no_xser(args, tp_rank, pp_rank):
    load_dir = os.path.join(args.input_dir, "dp_rank_00_tp_rank_{:02d}_pp_rank_{:02d}.pt".format(tp_rank, pp_rank)) #changed (removed checkpoint) but add dp rank as arg later TODO
    partial_state = torch.load(load_dir)
    return partial_state


def save_full(args, full_model):
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)
    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, "checkpoint.pt")
    print(f"Saving full checkpoint to {save_path}")
    torch.save(full_model, save_path)


def save_partial_xser(args, partial_state, tp_rank, pp_rank):
    save_dir = os.path.join(args.output_dir, "tp_rank_{:02d}_pp_rank_{:02d}".format(tp_rank, pp_rank))
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving to {save_dir}")
    xser.save(partial_state, save_dir)


def save_partial_no_xser(args, partial_state, tp_rank, pp_rank):
    save_dir = os.path.join(args.output_dir, "tp_rank_{:02d}_pp_rank_{:02d}".format(tp_rank, pp_rank))
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to {save_dir}")
    torch.save(partial_state, os.path.join(save_dir, "checkpoint.pt"))


# Convertion Entries
def convert_from_xser(args):
    for tp_rank in range(args.tp_size):
        for pp_rank in range(args.pp_size):
            partial_state = load_partial_xser(args, tp_rank, pp_rank)
            save_partial_no_xser(args, partial_state, tp_rank, pp_rank)


def convert_to_xser(args):
    for tp_rank in range(args.tp_size):
        for pp_rank in range(args.pp_size):
            partial_state = load_partial_no_xser(args, tp_rank, pp_rank)
            save_partial_xser(args, partial_state, tp_rank, pp_rank)


def convert_from_full_model(args):
    full_state = load_full(args)
    partitions = create_partition(args.n_layers, args.pp_size)
    print(f"pipeline_cuts {partitions}")
    if args.kv_size_multiplier > 1:
        assert args.config is not None, "If kv_size_multipler is greater than 1, need to pass config.json"
    for tp_rank in range(args.tp_size):
        for pp_rank in range(args.pp_size):
            partial_state = translate_llama_full_state_dict_to_tp(
                full_state,
                args.tp_size,
                tp_rank,
                args.pp_size,
                pp_rank,
                partitions,
                args.kv_size_multiplier,
                args.config,
            )
            if args.save_xser:
                save_partial_xser(args, partial_state, tp_rank, pp_rank)
            else:
                save_partial_no_xser(args, partial_state, tp_rank, pp_rank)


def convert_to_full_model(args):
    full_model = merge_llama_tp_checkpoints(args)
    save_full(args, full_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input model/weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save converted model/weights")
    parser.add_argument(
        "--model_key", type=str, default="model", help="Key of the model state dict in the checkpoint object"
    )
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor Parallel degree for the model")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline Parallel degree for the model")
    parser.add_argument("--n_layers", type=int, default=0, help="Number of Layers")
    parser.add_argument("--load_xser", type=bool, default=False, help="Load from xser saved checkpoints")
    parser.add_argument("--save_xser", type=bool, default=False, help="Save with xser")
    parser.add_argument(
        "--convert_from_xser", action="store_true", help="Convert xser saved checkpoint to normal torch checkpoint"
    )
    parser.add_argument(
        "--convert_to_xser", action="store_true", help="Convert normal torch checkpoint to xser checkpoint"
    )
    parser.add_argument("--convert_from_full_model", action="store_true", help="Convert full model to sharded model")
    parser.add_argument("--convert_to_full_model", action="store_true", help="Convert sharded model to full model")
    parser.add_argument(
        "--kv_size_multiplier", type=int, default=1, help="Factor by which the KV heads were replicated"
    )
    parser.add_argument("--config", type=str, default=None, help="Config.json")

    args, _ = parser.parse_known_args()
    if args.convert_from_full_model:
        convert_from_full_model(args)
    elif args.convert_to_full_model:
        convert_to_full_model(args)
    elif args.convert_from_xser:
        convert_from_xser(args)
    elif args.convert_to_xser:
        convert_to_xser(args)


# python3 convert_ckpt.py  --input_dir ./trn_ckpts/step_3500/model  --output_dir ./ckpt_pt/step_3500  --convert_to_full_model  --tp_size 8  --load_xser True
#python3 convert_ckpt.py  --input_dir ./trn_ckpts_mixed/step_2500/model  --output_dir ./ckpt_mix_pt/step_2500  --convert_to_full_model  --tp_size 8  --load_xser True
