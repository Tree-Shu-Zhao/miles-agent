"""configs/webshop.yaml must validate under Gym's server-config schema *offline*.

Gym refuses to start an "almost-server" (a block that has an entrypoint but fails
pydantic validation) and the whole gym boot dies with it. That only surfaced at
a real boot when `domain: web` was not a Gym Domain value, so
every server block is validated here, the same way global_config.py does it.
"""

from copy import deepcopy
from pathlib import Path

from nemo_gym.config_types import is_almost_server, maybe_get_server_instance_config
from omegaconf import OmegaConf

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "webshop.yaml"


def _server_blocks(cfg):
    return {name: block for name, block in cfg.items() if is_almost_server(block)}


def test_every_server_block_validates_under_gym_schema():
    blocks = _server_blocks(OmegaConf.load(CONFIG))
    assert blocks, f"no server blocks found in {CONFIG}"
    for name, block in blocks.items():
        _config, error = maybe_get_server_instance_config(name=name, server_type_config_dict=block)
        assert error is None, f"{name} in {CONFIG.name} fails Gym validation:\n{error}"


def test_validation_catches_a_bad_domain_value():
    # Regression guard for the check itself: the exact typo that killed that boot.
    cfg = deepcopy(OmegaConf.load(CONFIG))
    name = "webshop_env"
    cfg[name]["resources_servers"][name]["domain"] = "web"
    _config, error = maybe_get_server_instance_config(name=name, server_type_config_dict=cfg[name])
    assert error is not None and "domain" in str(error)
