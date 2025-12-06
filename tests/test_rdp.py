import torch

from project_psf.RDP import RDP, neighbor_product


def test_rdp_weights_without_kappa_match_voxel_weights():
    rdp = RDP((2, 2, 2))
    assert torch.allclose(rdp.weights, rdp._voxel_size_weights)


def test_rdp_num_anatomical_neighbors_zero_masks_all_weights():
    rdp = RDP((2, 2, 2), num_anatomical_neighbors=0)
    rdp.kappa = torch.ones(rdp.in_shape, dtype=torch.float32)
    assert torch.count_nonzero(rdp.weights) == 0


def test_rdp_anatomical_mask_respects_neighbor_count():
    rdp = RDP((3, 3, 3), num_anatomical_neighbors=5)
    kappa = torch.arange(27, dtype=torch.float32).reshape(rdp.in_shape)
    rdp.kappa = kappa

    mask = rdp._compute_anatomical_mask()
    assert mask is not None
    assert torch.all(mask.sum(dim=0) == 5)

    base_weights = (
        neighbor_product(rdp.kappa, padding=rdp._padding) * rdp._voxel_size_weights
    )
    assert torch.allclose(rdp.weights, base_weights * mask)


def test_rdp_setting_neighbor_count_updates_weights():
    rdp = RDP((3, 3, 3))
    rdp.kappa = torch.linspace(0.0, 1.0, steps=27, dtype=torch.float32).reshape(
        rdp.in_shape
    )
    original_weights = rdp.weights.clone()

    rdp.num_anatomical_neighbors = 4

    assert not torch.allclose(rdp.weights, original_weights)
    mask = rdp._compute_anatomical_mask()
    assert mask is not None
    assert torch.all(mask.sum(dim=0) == 4)
