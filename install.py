from sys import executable
import os
import subprocess
import platform
from typing import Union


def pip_install(package: Union[str, list]) -> None:
    """
    Efetua chamada para fazer instalar com pip o que foi recebido em package
    """
    cmd = ["pip", "install"]
    if type(package) == str:
        # formato necessario para subprocess
        package = package.split(" ")
    cmd.extend(package)
    subprocess.call(cmd, cwd=os.path.dirname(executable))


def py_torch_install(torch_version: str = "==1.3.0", vision_version: str = "==0.4.1", end="") -> None:
    pip_install([f"torch{torch_version}{end}",
                 f"torchvision{vision_version}{end}",
                 f"-f",
                 f"https://download.pytorch.org/whl/torch_stable.html"])


def linux_cuda(ver: str) -> bool:
    return os.path.exists(f"/usr/local/cuda-{ver}/bin")


def win_cuda(ver: str) -> bool:
    return os.path.exists(f"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v{ver}")


if __name__ == "__main__":
    op_sys = platform.system().lower()

    # Verifica se ha alguma instalacao do CUDA para permitir execucao do PyTorch com GPU, ou instala a versao para CPU
    if op_sys == "darwin" or linux_cuda("10.1"):
        py_torch_install("", "")

    elif win_cuda("9.2") or linux_cuda("9.2"):
        py_torch_install(end="'+cu92'")

    elif win_cuda("10.1"):
        py_torch_install()

    elif op_sys == "windows" or op_sys == "linux":
        if os.path.exists("/usr/local/cuda/version.txt") or (op_sys == "windows" and os.getenv("CUDA_PATH")):
            print("Instalando versao do PyTorch sem uso do CUDA toolkit, atualize seu CUDA para usa-lo no PyTorch")

        py_torch_install(end="'+cpu'")

    # Instalacao de requisitos ausentes
    pip_install(["-r", os.path.abspath("./requirements.txt")])
