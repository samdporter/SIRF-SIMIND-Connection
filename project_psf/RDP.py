from __future__ import annotations

import abc
from itertools import product
from typing import Any, Tuple

import torch
import torch.nn.functional as F


TensorLike = Any


def _as_tensor(
    x: TensorLike, device: torch.device | None, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Convert arbitrary input to a tensor on the requested device."""
    return torch.as_tensor(x, device=device, dtype=dtype)


def _with_padding(x: torch.Tensor, padding: str) -> torch.Tensor:
    """Pad by one voxel in every direction using the requested mode."""
    if padding in {"replicate", "edge"}:
        x_padded = x
        for dim in range(x.ndim):
            front = x_padded.select(dim, 0).unsqueeze(dim)
            back = x_padded.select(dim, -1).unsqueeze(dim)
            x_padded = torch.cat([front, x_padded, back], dim=dim)
        return x_padded

    pad = (1,) * (2 * x.ndim)
    return F.pad(x, pad, mode=padding)


def neighbor_difference_and_sum(
    x: torch.Tensor, padding: str = "replicate"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differences and sums with nearest neighbors for n-D arrays."""
    x_padded = _with_padding(x, padding)
    num_neigh = 3**x.ndim - 1

    d = torch.zeros((num_neigh,) + x.shape, dtype=x.dtype, device=x.device)
    s = torch.zeros_like(d)

    center = num_neigh // 2
    for i, ind in enumerate(product(range(3), repeat=x.ndim)):
        if i == center:
            continue

        slices = []
        for j in ind:
            if j - 2 < 0:
                slices.append(slice(j, j - 2))
            else:
                slices.append(slice(j, None))
        sl = tuple(slices)

        target = i if i < center else i - 1
        d[target, ...] = x - x_padded[sl]
        s[target, ...] = x + x_padded[sl]

    return d, s


def neighbor_product(x: torch.Tensor, padding: str = "replicate") -> torch.Tensor:
    """Return forward/backward neighbor products along each dimension."""
    x_padded = _with_padding(x, padding)
    num_neigh = 3**x.ndim - 1

    p = torch.zeros((num_neigh,) + x.shape, dtype=x.dtype, device=x.device)

    center = num_neigh // 2
    for i, ind in enumerate(product(range(3), repeat=x.ndim)):
        if i == center:
            continue

        slices = []
        for j in ind:
            if j - 2 < 0:
                slices.append(slice(j, j - 2))
            else:
                slices.append(slice(j, None))
        sl = tuple(slices)

        target = i if i < center else i - 1
        p[target, ...] = x * x_padded[sl]

    return p


class SmoothFunction(abc.ABC):
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        device: torch.device | None = None,
        scale: float = 1.0,
    ) -> None:
        self._in_shape = in_shape
        self._scale = scale
        self._device = device or torch.device("cpu")

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, scale: float) -> None:
        self._scale = scale

    @property
    def in_shape(self) -> Tuple[int, ...]:
        return self._in_shape

    @property
    def device(self) -> torch.device:
        return self._device

    @abc.abstractmethod
    def _call(self, x: torch.Tensor) -> torch.Tensor | float:
        raise NotImplementedError

    @abc.abstractmethod
    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _prepare_input(
        self, x: TensorLike
    ) -> tuple[torch.Tensor, bool, bool, torch.device | None]:
        input_is_tensor = isinstance(x, torch.Tensor)
        input_device = x.device if input_is_tensor else None

        tensor = _as_tensor(x, device=self._device)
        flat_input = tensor.ndim == 1
        if flat_input:
            tensor = tensor.reshape(self._in_shape)

        return tensor, flat_input, input_is_tensor, input_device

    def _restore_output(
        self,
        output: torch.Tensor,
        flat_input: bool,
        input_is_tensor: bool,
        input_device: torch.device | None,
    ) -> torch.Tensor | Any:
        if flat_input:
            output = output.reshape(-1)
        if input_is_tensor:
            return output.to(device=input_device)
        return output.cpu().numpy()

    def __call__(self, x: TensorLike) -> float:
        tensor, flat_input, _, _ = self._prepare_input(x)
        result = self._call(tensor)
        if self._scale != 1.0:
            result = self._scale * result
        if isinstance(result, torch.Tensor):
            result = result.item()
        return float(result)

    def gradient(self, x: TensorLike) -> torch.Tensor | Any:
        tensor, flat_input, input_is_tensor, input_device = self._prepare_input(x)
        result = self._gradient(tensor)
        if self._scale != 1.0:
            result = self._scale * result
        return self._restore_output(result, flat_input, input_is_tensor, input_device)

    def prox_function(self, z: TensorLike, x: TensorLike, T: TensorLike) -> float:
        z_t = _as_tensor(z, device=self._device)
        x_t = _as_tensor(x, device=self._device)
        T_t = _as_tensor(T, device=self._device)
        return float(self(z_t) + 0.5 * torch.sum(((z_t - x_t) ** 2) / T_t).item())

    def prox_gradient(
        self, z: TensorLike, x: TensorLike, T: TensorLike
    ) -> torch.Tensor | Any:
        z_t = _as_tensor(z, device=self._device)
        x_t = _as_tensor(x, device=self._device)
        T_t = _as_tensor(T, device=self._device)
        base_grad = (
            self._gradient(z_t)
            if self._scale == 1.0
            else self._scale * self._gradient(z_t)
        )
        grad = base_grad + (z_t - x_t) / T_t
        _, flat_input, input_is_tensor, input_device = self._prepare_input(z)
        return self._restore_output(grad, flat_input, input_is_tensor, input_device)


class SmoothFunctionWithDiagonalHessian(SmoothFunction):
    @abc.abstractmethod
    def _diag_hessian(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def diag_hessian(self, x: TensorLike) -> torch.Tensor | Any:
        tensor, flat_input, input_is_tensor, input_device = self._prepare_input(x)
        result = self._diag_hessian(tensor)
        if self._scale != 1.0:
            result = self._scale * result
        return self._restore_output(result, flat_input, input_is_tensor, input_device)

    def inv_diag_hessian(
        self, x: TensorLike, epsilon: float = 1e-6
    ) -> torch.Tensor | Any:
        tensor, flat_input, input_is_tensor, input_device = self._prepare_input(x)
        result = self._inv_diag_hessian(tensor, epsilon)
        if self._scale != 1.0:
            result = result / self._scale
        return self._restore_output(result, flat_input, input_is_tensor, input_device)


class RDP(SmoothFunctionWithDiagonalHessian):
    def __init__(
        self,
        in_shape: Tuple[int, ...],
        voxel_size: TensorLike | None = None,
        eps: float | None = None,
        gamma: float = 2.0,
        padding: str = "replicate",
        device: torch.device | None = None,
    ) -> None:
        self._gamma = gamma
        self._eps = eps if eps is not None else torch.finfo(torch.float32).eps
        self._padding = padding
        self._ndim = len(in_shape)
        self._device = device or torch.device("cpu")

        super().__init__(in_shape=in_shape, device=self._device)

        self._num_neigh = 3**self._ndim - 1
        voxel_size = voxel_size if voxel_size is not None else (1.0,) * self._ndim
        self._voxel_size = _as_tensor(voxel_size, device=self._device)

        self._voxel_size_weights = torch.zeros(
            (self._num_neigh,) + in_shape,
            dtype=self._voxel_size.dtype,
            device=self._device,
        )

        center = self._num_neigh // 2
        for i, ind in enumerate(product(range(3), repeat=self._ndim)):
            if i == center:
                continue

            offset = (
                _as_tensor(ind, device=self._device, dtype=self._voxel_size.dtype) - 1
            )
            vw = self._voxel_size[-1] / torch.linalg.norm(offset * self._voxel_size)

            target = i if i < center else i - 1
            self._voxel_size_weights[target, ...] = vw

        self._weights = self._voxel_size_weights
        self._kappa = None

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    @property
    def kappa(self) -> torch.Tensor | None:
        return self._kappa

    @kappa.setter
    def kappa(self, image: TensorLike) -> None:
        self._kappa = _as_tensor(image, device=self._device)
        self._weights = (
            neighbor_product(self._kappa, padding=self._padding)
            * self._voxel_size_weights
        )

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_tensor(x, device=self._device)
        if float(torch.min(x)) < 0:
            return torch.tensor(torch.inf, device=self._device)

        d, s = neighbor_difference_and_sum(x, padding=self._padding)
        phi = s + self.gamma * torch.abs(d) + self.eps

        tmp = (d**2) / phi
        if self._weights is not None:
            tmp = tmp * self._weights

        return 0.5 * torch.sum(tmp, dtype=torch.float64)

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_tensor(x, device=self._device)
        d, s = neighbor_difference_and_sum(x, padding=self._padding)
        phi = s + self.gamma * torch.abs(d) + self.eps

        tmp = d * (2 * phi - (d + self.gamma * torch.abs(d))) / (phi**2)
        if self._weights is not None:
            tmp = tmp * self._weights

        return tmp.sum(dim=0)

    def _diag_hessian(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_tensor(x, device=self._device)
        d, s = neighbor_difference_and_sum(x, padding=self._padding)
        phi = s + self.gamma * torch.abs(d) + self.eps
        phi = torch.clamp(phi, min=self.eps)

        tmp = ((s - d + self.eps) ** 2) / (phi**3)
        tmp = torch.nan_to_num(tmp, nan=0.0, posinf=0.0, neginf=0.0)
        if self._weights is not None:
            tmp = tmp * self._weights

        return 2 * tmp.sum(dim=0)

    def _inv_diag_hessian(self, x: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        input_is_tensor = isinstance(x, torch.Tensor)
        diag_hess = self._diag_hessian(x).abs()
        denom = max(epsilon, 1e-12)
        result = 1.0 / torch.clamp(diag_hess, min=denom)
        if input_is_tensor:
            return result
        return result.cpu().numpy()
