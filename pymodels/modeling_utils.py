import os
from typing import Dict, Union
import time
import numpy as np
import ml_dtypes
from safetensors import safe_open
import glob
from tqdm import tqdm
import infinicore



def infini_to_numpy(infini_tensor: infinicore.Tensor):
    def infini_to_ctype_dtype(infini_dtype):
        import ctypes

        if infini_dtype == infinicore.int32:
            return ctypes.c_int32
        elif infini_dtype == infinicore.float32:
            return ctypes.c_float
        elif infini_dtype == infinicore.int64:
            return ctypes.c_int64
        else:
            raise ValueError(f"Unsupported py_dtype: {infini_dtype}")

    infini_tensor_cpu = infinicore.empty_like(infini_tensor, device=infinicore.device("cpu"))
    infini_tensor_cpu.copy_(infini_tensor)

    # 获取数据指针和形状信息
    data_ptr = infini_tensor_cpu.data_ptr()
    num_elements = infini_tensor_cpu.numel()
    original_shape = infini_tensor_cpu.shape

    # 创建1D NumPy数组（共享内存）
    ArrayType = infini_to_ctype_dtype(infini_tensor_cpu.dtype) * num_elements
    array = ArrayType.from_address(data_ptr)
    np_flat = np.ctypeslib.as_array(array)

    # 重塑为原始形状
    np_array = np_flat.reshape(original_shape)

    return np.copy(np_array)



def check_parameters(model_keys: list, already_loaded_keys: list):
    model_keys = set(model_keys)
    already_loaded_keys = set(already_loaded_keys)
    intersection = model_keys & already_loaded_keys

    missing_keys = model_keys - intersection
    unexpected_keys = already_loaded_keys - intersection
    error_msgs: list[str] = []

    if len(unexpected_keys) > 0:
        error_msgs.append(
            "Unexpected key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in unexpected_keys)
            )
        )
    if len(missing_keys) > 0:
        error_msgs.append(
            "Missing key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in missing_keys)
            )
        )

    if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict\n\t{}".format("\n\t".join(error_msgs))
        )


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike],  dtype=ml_dtypes.bfloat16
) -> Dict[str, np.ndarray]:
    """
    Reads a `safetensor` checkpoint file. We load the checkpoint on "cpu" by default.
    """
    if not checkpoint_file.endswith(".safetensors"):
        return {}

    state_dict = {}
    with safe_open(checkpoint_file, framework="np") as f:
        metadata = f.metadata()
        if metadata is not None and metadata.get("format") not in [
            "pt",
            "tf",
            "flax",
            "mlx",
        ]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata."
            )

        for k in f.keys():
            state_dict[k] = f.get_tensor(k).astype(dtype)

    return state_dict



def load_model_state_dict_by_file(
    model: infinicore.nn.Module,
    model_path: str,
    dtype=infinicore.dtype,
) -> Dict[str, infinicore.Tensor]:
    """
    Load the model weights from file.
    """
    print(" load weights ......")
    t1 = time.time()

    np_dtype = infinicore.utils.infinicore_to_numpy_dtype(dtype)
    model_keys = model.state_dict().keys()


    already_loaded_keys = []
    file_list = glob.glob(os.path.join(model_path, "*.safetensors"))
  
    for file_path in tqdm(file_list, desc="Processing files"):
        tqdm.write(f"Processing: {os.path.basename(file_path)}")

        # --------------------------------------------------------- #
        #          Load weights from *.safetensors file
        # --------------------------------------------------------- #
        model_param = load_state_dict(file_path, dtype=np_dtype)
        already_loaded_keys.extend(model_param.keys())

        # --------------------------------------------------------- #
        #         model_param_infini references torch.Tensor
        # --------------------------------------------------------- #
        model_param_infini = {}
        for key in model_param.keys():
            model_param_infini[key] = infinicore.from_numpy(model_param[key])

        model.load_state_dict(model_param_infini, strict=False)
        infinicore.sync_device()


    check_parameters(model_keys, already_loaded_keys)

    t2 = time.time()
    print(f" load weights over! {(t2 - t1) * 1000} ms \n")